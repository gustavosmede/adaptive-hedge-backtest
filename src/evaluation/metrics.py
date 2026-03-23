from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def compute_metrics(df: pd.DataFrame, annual_factor: int) -> dict[str, float]:
    rets = df["net_return"].fillna(0.0)
    gross = df["gross_return"].fillna(0.0)

    mean_ret = float(rets.mean())
    std_ret = float(rets.std(ddof=0))

    downside = rets.copy()
    downside[downside > 0] = 0.0
    downside_std = float(np.sqrt((downside.pow(2).mean())))

    final_equity = float(df["equity"].iloc[-1])
    start_equity = float(df["equity"].iloc[0])

    n = max(len(df), 1)
    cagr = (final_equity / start_equity) ** (annual_factor / n) - 1.0 if start_equity > 0 else 0.0

    drawdown = df["drawdown"].fillna(0.0)
    max_dd = float(drawdown.min())

    sharpe = _safe_div(mean_ret * np.sqrt(annual_factor), std_ret)
    sortino = _safe_div(mean_ret * np.sqrt(annual_factor), downside_std)
    calmar = _safe_div(cagr, abs(max_dd))

    pnl_gross = float(gross.sum())
    pnl_net = float(rets.sum())
    turnover = float(df["turnover"].sum())
    avg_exposure = float(df["position"].abs().mean())

    non_zero = max(int((rets != 0).sum()), 1)
    win_rate = float((rets > 0).sum() / non_zero)

    return {
        "final_equity": final_equity,
        "pnl_gross_sum": pnl_gross,
        "pnl_net_sum": pnl_net,
        "cagr": float(cagr),
        "max_drawdown": max_dd,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "turnover": turnover,
        "avg_exposure_abs": avg_exposure,
        "total_trade_cost": float(df["trade_cost"].sum()),
        "win_rate": win_rate,
    }


def summarize_metrics(metrics_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_map).T.sort_values("sharpe", ascending=False)
