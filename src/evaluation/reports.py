from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_strategy_outputs(
    strategy_name: str,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    out_dir: Path,
) -> None:
    bars.to_csv(out_dir / f"{strategy_name}_bars.csv")
    if trades.empty:
        pd.DataFrame(columns=["timestamp", "position_prev", "position_new", "delta", "close", "trade_cost", "equity_after", "wf_fold"]).to_csv(
            out_dir / f"{strategy_name}_trades.csv", index=False
        )
    else:
        trades.to_csv(out_dir / f"{strategy_name}_trades.csv")


def plot_price(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="BTCUSDT Close", linewidth=1.0)
    ax.set_title("BTCUSDT Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_equity_comparison(
    strategy_bars: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, bars in strategy_bars.items():
        ax.plot(bars.index, bars["equity"], label=name)
    ax.set_title("Equity Curve Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown_comparison(
    strategy_bars: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, bars in strategy_bars.items():
        ax.plot(bars.index, bars["drawdown"], label=name)
    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_table(metrics_df: pd.DataFrame, out_path: Path) -> None:
    metrics_df.to_csv(out_path)
