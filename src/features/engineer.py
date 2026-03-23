from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=0)
    return mu / (sigma.replace(0.0, np.nan))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    log_ret = np.log(df["close"]).diff()
    df["rv_short"] = log_ret.rolling(24).std(ddof=0)
    df["rv_long"] = log_ret.rolling(96).std(ddof=0)

    df["rolling_sharpe"] = _rolling_sharpe(df["ret_1"], window=48)

    df["ma_short"] = df["close"].rolling(24).mean()
    df["ma_long"] = df["close"].rolling(96).mean()
    df["ma_dist"] = (df["close"] / df["ma_long"]) - 1.0

    rolling_std = df["close"].rolling(96).std(ddof=0)
    df["zscore_price"] = (df["close"] - df["ma_long"]) / rolling_std.replace(0.0, np.nan)

    df["momentum"] = df["close"].pct_change(12)
    df["atr"] = _atr(df, window=14)
    df["atr_pct"] = df["atr"] / df["close"]

    vol_mean = df["volume"].rolling(96).mean()
    vol_std = df["volume"].rolling(96).std(ddof=0)
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    df["trend_regime"] = np.where(df["ma_short"] > df["ma_long"], 1.0, -1.0)

    rolling_peak = df["close"].cummax()
    df["price_drawdown"] = (df["close"] / rolling_peak) - 1.0

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "rv_short",
        "rv_long",
        "rolling_sharpe",
        "ma_dist",
        "zscore_price",
        "momentum",
        "atr_pct",
        "volume_z",
        "trend_regime",
        "price_drawdown",
    ]

    df = df.dropna(subset=feature_cols).copy()
    return df
