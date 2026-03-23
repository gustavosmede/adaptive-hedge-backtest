from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyState


class BuyAndHoldStrategy(BaseStrategy):
    name = "buy_and_hold"

    def __init__(self, target: float = 1.0) -> None:
        self.target = target

    def decide(self, row: pd.Series, state: StrategyState) -> float:
        return float(self.target)


class RuleBasedDynamicStrategy(BaseStrategy):
    name = "rule_based_dynamic"

    def __init__(self, allow_short: bool = True) -> None:
        self.allow_short = allow_short

    def decide(self, row: pd.Series, state: StrategyState) -> float:
        trend = float(row.get("trend_regime", 0.0))
        momentum = float(row.get("momentum", 0.0))
        rv_short = float(row.get("rv_short", 0.0))
        rv_long = float(row.get("rv_long", 0.0))
        drawdown = float(state.current_drawdown)

        vol_ratio = rv_short / rv_long if rv_long > 1e-12 else 1.0
        base = 0.5

        if trend > 0 and momentum > 0:
            base += 0.35
        elif trend < 0 and momentum < 0:
            base -= 0.35

        if vol_ratio > 1.25:
            base -= 0.25
        if drawdown < -0.10:
            base -= 0.20

        if not self.allow_short:
            return float(np.clip(base, 0.0, 1.0))

        return float(np.clip((base - 0.5) * 2.0, -1.0, 1.0))
