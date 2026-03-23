from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from config import StrategyConfig
from src.strategies.base import BaseStrategy, StrategyState


@dataclass(slots=True)
class CalibratedParams:
    trend_weight: float
    momentum_weight: float
    mean_reversion_weight: float
    vol_penalty_weight: float
    drawdown_penalty_weight: float
    cost_aversion: float
    hysteresis_band: float
    vol_target: float
    bull_mode_boost: float


class AdaptiveHedgeStrategy(BaseStrategy):
    name = "adaptive_hedge"

    def __init__(self, cfg: StrategyConfig, allow_short: bool = False, execution_cost: float = 0.0006) -> None:
        self.cfg = cfg
        self.allow_short = allow_short
        self.execution_cost = execution_cost
        self.levels = (
            np.array(cfg.rebalance_levels_long_short)
            if allow_short
            else np.array(cfg.rebalance_levels_long_only)
        )
        self.feature_cols = [
            "ret_3",
            "ret_6",
            "ret_12",
            "rv_short",
            "rv_long",
            "rolling_sharpe",
            "ma_dist",
            "zscore_price",
            "momentum",
            "atr_pct",
            "volume_z",
        ]
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.params_ = CalibratedParams(
            trend_weight=cfg.trend_weight,
            momentum_weight=cfg.momentum_weight,
            mean_reversion_weight=cfg.mean_reversion_weight,
            vol_penalty_weight=cfg.vol_penalty_weight,
            drawdown_penalty_weight=cfg.drawdown_penalty_weight,
            cost_aversion=cfg.cost_aversion,
            hysteresis_band=cfg.hysteresis_band,
            vol_target=cfg.vol_target,
            bull_mode_boost=0.20,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        x = train_df[self.feature_cols].copy()
        self.mean_ = x.mean()
        self.std_ = x.std(ddof=0).replace(0.0, 1.0)
        calibration_df = train_df.iloc[::4].copy()
        self.params_ = self._calibrate(calibration_df)

    def _z(self, row: pd.Series, col: str) -> float:
        if self.mean_ is None or self.std_ is None:
            return float(row.get(col, 0.0))
        return float((row.get(col, 0.0) - self.mean_[col]) / self.std_[col])

    def _base_signal(self, row: pd.Series, params: CalibratedParams, current_drawdown: float) -> float:
        trend_component = params.trend_weight * (
            0.7 * row.get("trend_regime", 0.0) + 0.3 * np.tanh(self._z(row, "ma_dist"))
        )
        momentum_component = params.momentum_weight * np.tanh(
            self._z(row, "momentum") + 0.5 * self._z(row, "ret_6")
        )
        mean_reversion_component = -params.mean_reversion_weight * np.tanh(self._z(row, "zscore_price"))

        vol_penalty = params.vol_penalty_weight * max(0.0, self._z(row, "rv_short"))
        dd_penalty = params.drawdown_penalty_weight * abs(min(0.0, current_drawdown))

        gross_score = trend_component + momentum_component + mean_reversion_component
        risk_adjusted = gross_score - self.cfg.risk_aversion * (vol_penalty + dd_penalty)

        realized_vol = max(float(row.get("rv_short", 0.0)), 1e-6)
        vol_scaled = risk_adjusted * min(1.0, params.vol_target / realized_vol)
        continuous_target = float(np.tanh(np.clip(vol_scaled, -2.5, 2.5)))

        if not self.allow_short:
            continuous_target = float(np.clip((continuous_target + 1.0) / 2.0, 0.0, 1.0))

        return continuous_target

    def _apply_bull_override(
        self,
        row: pd.Series,
        base_target: float,
        current_drawdown: float,
        params: CalibratedParams,
    ) -> float:
        if self.allow_short:
            return base_target

        trend_ok = float(row.get("trend_regime", 0.0)) > 0.0
        ma_ok = self._z(row, "ma_dist") > -0.15
        momentum_ok = self._z(row, "momentum") > -0.10 and self._z(row, "ret_6") > -0.10
        vol_ok = float(row.get("rv_short", 0.0)) <= max(float(row.get("rv_long", 0.0)), 1e-6) * 1.15
        dd_ok = current_drawdown > -0.12

        if trend_ok and ma_ok and momentum_ok and vol_ok and dd_ok:
            boosted = max(base_target, 0.50 + params.bull_mode_boost)
            if self._z(row, "momentum") > 0.5 and self._z(row, "ma_dist") > 0.25:
                boosted = max(boosted, 1.0)
            elif self._z(row, "momentum") > 0.0:
                boosted = max(boosted, 0.75)
            return float(np.clip(boosted, 0.0, 1.0))

        return base_target

    def _candidate_penalty(
        self,
        candidate: float,
        row: pd.Series,
        current_position: float,
        current_drawdown: float,
        params: CalibratedParams,
    ) -> float:
        trade_penalty = params.cost_aversion * abs(candidate - current_position)
        risk_penalty = 0.5 * max(0.0, self._z(row, "rv_short")) * candidate
        dd_penalty = 0.75 * abs(min(0.0, current_drawdown)) * candidate
        return float(trade_penalty + risk_penalty + dd_penalty)

    def _select_target(
        self,
        row: pd.Series,
        current_position: float,
        current_drawdown: float,
        params: CalibratedParams,
    ) -> float:
        base_target = self._base_signal(row, params, current_drawdown)
        base_target = self._apply_bull_override(row, base_target, current_drawdown, params)
        best_target = float(current_position)
        best_score = -np.inf

        for candidate in self.levels:
            candidate = float(candidate)
            attraction = -abs(candidate - base_target)
            score = attraction - self._candidate_penalty(
                candidate=candidate,
                row=row,
                current_position=current_position,
                current_drawdown=current_drawdown,
                params=params,
            )
            if score > best_score:
                best_score = score
                best_target = candidate

        if abs(best_target - current_position) < params.hysteresis_band:
            return float(current_position)

        return float(best_target)

    def _simulate_train_objective(self, train_df: pd.DataFrame, params: CalibratedParams) -> float:
        prev_position = 0.0
        prev_equity = 1.0
        peak_equity = 1.0
        total_turnover = 0.0
        net_returns: list[float] = []
        exposure_path: list[float] = []

        close_ret = train_df["close"].pct_change().fillna(0.0).to_numpy()

        for i, (_, row) in enumerate(train_df.iterrows()):
            current_dd = (prev_equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
            desired = self._select_target(
                row=row,
                current_position=prev_position,
                current_drawdown=current_dd,
                params=params,
            )

            delta = abs(desired - prev_position)
            trade_cost = delta * self.execution_cost
            gross = desired * close_ret[i]
            net = gross - trade_cost
            prev_equity *= 1.0 + net
            peak_equity = max(peak_equity, prev_equity)
            prev_position = desired
            total_turnover += delta
            net_returns.append(net)
            exposure_path.append(abs(desired))

        returns = np.asarray(net_returns, dtype=float)
        if returns.size == 0:
            return -np.inf

        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        sharpe_like = mean_ret / std_ret if std_ret > 1e-12 else -np.inf
        max_dd = min(0.0, (prev_equity / peak_equity) - 1.0)
        total_return = prev_equity - 1.0
        avg_exposure = float(np.mean(exposure_path)) if exposure_path else 0.0

        return float(
            0.55 * sharpe_like
            + 0.35 * total_return
            + 0.08 * avg_exposure
            - 0.02 * total_turnover
            + 0.35 * max_dd
        )

    def _calibrate(self, train_df: pd.DataFrame) -> CalibratedParams:
        grid = product(
            (0.55, 0.85),
            (0.35, 0.65),
            (0.10, 0.25),
            (0.70, 1.10),
            (0.80, 1.20),
            (0.90, 1.30),
            (0.20, 0.35),
            (0.018, 0.028),
            (0.15, 0.30),
        )

        best_params = self.params_
        best_score = -np.inf

        for values in grid:
            candidate = CalibratedParams(*values)
            score = self._simulate_train_objective(train_df, candidate)
            if score > best_score:
                best_score = score
                best_params = candidate

        return best_params

    def decide(self, row: pd.Series, state: StrategyState) -> float:
        return self._select_target(
            row=row,
            current_position=state.current_position,
            current_drawdown=state.current_drawdown,
            params=self.params_,
        )
