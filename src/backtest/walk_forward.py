from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestInitState, BacktestOutput, BacktestParams, run_backtest
from src.strategies.base import BaseStrategy


@dataclass(slots=True)
class WalkForwardParams:
    train_bars: int
    test_bars: int
    step_bars: int


@dataclass(slots=True)
class StrategyWalkForwardResult:
    bars: pd.DataFrame
    trades: pd.DataFrame


def _iter_splits(n: int, p: WalkForwardParams):
    start = 0
    while True:
        train_start = start
        train_end = train_start + p.train_bars
        test_end = train_end + p.test_bars

        if test_end > n:
            break

        yield train_start, train_end, train_end, test_end
        start += p.step_bars


def run_walk_forward(
    df: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    bt_params: BacktestParams,
    wf_params: WalkForwardParams,
) -> dict[str, StrategyWalkForwardResult]:
    if len(df) < (wf_params.train_bars + wf_params.test_bars):
        raise ValueError("Dados insuficientes para a configuração de walk-forward")

    outputs: dict[str, list[pd.DataFrame]] = {name: [] for name in strategies}
    trades: dict[str, list[pd.DataFrame]] = {name: [] for name in strategies}

    state_map = {
        name: BacktestInitState(
            initial_equity=bt_params.initial_capital,
            initial_position=0.0,
            cumulative_cost=0.0,
        )
        for name in strategies
    }

    for split_id, (tr_s, tr_e, te_s, te_e) in enumerate(_iter_splits(len(df), wf_params), start=1):
        train_df = df.iloc[tr_s:tr_e].copy()
        test_df = df.iloc[te_s:te_e].copy()

        for name, strategy in strategies.items():
            strategy.fit(train_df)
            bt: BacktestOutput = run_backtest(
                test_df,
                strategy,
                bt_params,
                init_state=state_map[name],
            )

            fold_bars = bt.bars.copy()
            fold_bars["wf_fold"] = split_id
            outputs[name].append(fold_bars)

            if not bt.trades.empty:
                fold_trades = bt.trades.copy()
                fold_trades["wf_fold"] = split_id
                trades[name].append(fold_trades)

            state_map[name] = BacktestInitState(
                initial_equity=bt.end_equity,
                initial_position=bt.end_position,
                cumulative_cost=bt.cumulative_cost,
            )

    result: dict[str, StrategyWalkForwardResult] = {}
    for name in strategies:
        bars_df = pd.concat(outputs[name]).sort_index()
        trades_df = pd.concat(trades[name]).sort_index() if trades[name] else pd.DataFrame()
        result[name] = StrategyWalkForwardResult(bars=bars_df, trades=trades_df)

    return result
