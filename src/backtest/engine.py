from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyState


@dataclass(slots=True)
class BacktestParams:
    transaction_cost: float
    slippage: float
    initial_capital: float
    min_holding_bars: int = 0
    rebalance_every_n_bars: int = 1
    max_position_step: float = 1.0


@dataclass(slots=True)
class BacktestInitState:
    initial_equity: float
    initial_position: float
    cumulative_cost: float


@dataclass(slots=True)
class BacktestOutput:
    bars: pd.DataFrame
    trades: pd.DataFrame
    end_equity: float
    end_position: float
    cumulative_cost: float


def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    params: BacktestParams,
    init_state: BacktestInitState | None = None,
) -> BacktestOutput:
    if df.empty:
        raise ValueError("DataFrame de backtest vazio")

    index = df.index
    n = len(df)

    if init_state is None:
        init_state = BacktestInitState(
            initial_equity=params.initial_capital,
            initial_position=0.0,
            cumulative_cost=0.0,
        )

    desired_position = np.zeros(n)
    active_position = np.zeros(n)
    gross_ret = np.zeros(n)
    trade_cost = np.zeros(n)
    net_ret = np.zeros(n)
    equity = np.zeros(n)
    running_drawdown = np.zeros(n)
    turnover = np.zeros(n)

    prev_desired = init_state.initial_position
    prev_active = init_state.initial_position
    prev_equity = init_state.initial_equity
    cum_cost = init_state.cumulative_cost
    peak_equity = prev_equity
    last_rebalance_bar = -10**9

    close_ret = df["close"].pct_change().fillna(0.0).values
    trades: list[dict] = []

    for i, (ts, row) in enumerate(df.iterrows()):
        current_dd = (prev_equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
        state = StrategyState(
            current_position=float(prev_active),
            previous_signal=float(prev_desired),
            cumulative_cost=float(cum_cost),
            current_drawdown=float(current_dd),
            bar_index=i,
        )

        desired = float(strategy.decide(row, state))
        desired = float(np.clip(desired, -1.0, 1.0))

        # Regras operacionais para conter churn.
        if (i % max(params.rebalance_every_n_bars, 1)) != 0:
            desired = float(prev_desired)
        elif (i - last_rebalance_bar) < params.min_holding_bars:
            desired = float(prev_desired)
        else:
            delta_desired = desired - prev_desired
            if abs(delta_desired) > params.max_position_step:
                desired = float(prev_desired + np.sign(delta_desired) * params.max_position_step)

        desired_position[i] = desired

        # execução na barra seguinte -> posição ativa hoje foi sinal de ontem
        active = prev_desired
        active_position[i] = active

        delta = abs(active - prev_active)
        tc = delta * (params.transaction_cost + params.slippage)
        trade_cost[i] = tc
        turnover[i] = delta
        cum_cost += tc

        gross = active * close_ret[i]
        gross_ret[i] = gross
        net = gross - tc
        net_ret[i] = net

        eq = prev_equity * (1.0 + net)
        equity[i] = eq

        peak_equity = max(peak_equity, eq)
        running_drawdown[i] = (eq / peak_equity) - 1.0

        if delta > 0:
            last_rebalance_bar = i
            trades.append(
                {
                    "timestamp": ts,
                    "position_prev": prev_active,
                    "position_new": active,
                    "delta": active - prev_active,
                    "close": float(row["close"]),
                    "trade_cost": tc,
                    "equity_after": eq,
                }
            )

        prev_desired = desired
        prev_active = active
        prev_equity = eq

    out = df.copy()
    out["desired_position"] = desired_position
    out["position"] = active_position
    out["position_change"] = pd.Series(active_position, index=index).diff().fillna(active_position[0])
    out["gross_return"] = gross_ret
    out["trade_cost"] = trade_cost
    out["net_return"] = net_ret
    out["turnover"] = turnover
    out["equity"] = equity
    out["drawdown"] = running_drawdown
    out["cumulative_cost"] = np.cumsum(trade_cost) + init_state.cumulative_cost

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.set_index("timestamp").sort_index()

    return BacktestOutput(
        bars=out,
        trades=trades_df,
        end_equity=float(equity[-1]),
        end_position=float(active_position[-1]),
        cumulative_cost=float(cum_cost),
    )
