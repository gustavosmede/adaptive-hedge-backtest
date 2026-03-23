from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class StrategyState:
    current_position: float
    previous_signal: float
    cumulative_cost: float
    current_drawdown: float
    bar_index: int


class BaseStrategy:
    name: str = "base"

    def fit(self, train_df: pd.DataFrame) -> None:
        return

    def reset(self) -> None:
        return

    def decide(self, row: pd.Series, state: StrategyState) -> float:
        raise NotImplementedError
