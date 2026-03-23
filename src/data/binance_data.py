from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from binance.client import Client

VALID_INTERVALS = {"5m", "15m", "30m", "1h"}


def _to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _cache_path(
    data_dir: Path,
    symbol: str,
    market: str,
    interval: str,
    start_date: str,
    end_date: str,
    cache_format: str,
) -> Path:
    filename = (
        f"{market}_{symbol}_{interval}_{start_date.replace('-', '')}_{end_date.replace('-', '')}."
        f"{cache_format}"
    )
    return data_dir / filename


def load_local_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de cache não encontrado: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato não suportado: {path.suffix}")

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    return df.sort_index()


def _save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_store = df.reset_index()
    if path.suffix == ".parquet":
        try:
            to_store.to_parquet(path, index=False)
            return
        except Exception:
            fallback = path.with_suffix(".csv")
            to_store.to_csv(fallback, index=False)
            return
    to_store.to_csv(path, index=False)


def _client() -> Client:
    # Endpoints públicos não exigem chave para klines históricos.
    return Client(api_key=None, api_secret=None)


def download_binance_klines(
    symbol: str,
    market: str,
    interval: str,
    start_date: str,
    end_date: str,
    data_dir: Path,
    cache_format: str = "parquet",
    use_cache: bool = True,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Intervalo não suportado: {interval}")
    if market not in {"spot", "futures"}:
        raise ValueError("market deve ser 'spot' ou 'futures'")

    cache_path = _cache_path(
        data_dir=data_dir,
        symbol=symbol,
        market=market,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        cache_format=cache_format,
    )

    if use_cache and cache_path.exists():
        return load_local_data(cache_path)

    # fallback para csv caso parquet esperado não exista, mas csv exista
    csv_fallback = cache_path.with_suffix(".csv")
    if use_cache and csv_fallback.exists():
        return load_local_data(csv_fallback)

    client = _client()
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)

    all_rows: list[list] = []
    current_start = start_ms
    limit = 1000

    while current_start < end_ms:
        if market == "spot":
            rows = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ms,
                limit=limit,
            )
        else:
            rows = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ms,
                limit=limit,
            )

        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        current_start = last_open_time + 1
        time.sleep(sleep_seconds)

        if len(rows) < limit:
            break

    if not all_rows:
        raise RuntimeError("Nenhum dado retornado pela Binance para os parâmetros selecionados")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[(df.index >= pd.Timestamp(start_date, tz="UTC")) & (df.index <= pd.Timestamp(end_date, tz="UTC"))]
    df = df[~df.index.duplicated(keep="first")]

    _save_data(df, cache_path)
    return df


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em OHLCV: {missing}")

    out = df[needed].copy()
    out = out.sort_index()
    out = out.astype(float)
    out = out[~out.index.duplicated(keep="first")]
    return out


def validate_data(df: pd.DataFrame, interval: str) -> None:
    if df.empty:
        raise ValueError("DataFrame vazio")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Índice temporal não está ordenado")
    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("Dados com NaN nas colunas OHLCV")

    expected_freq = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
    }[interval]

    inferred = pd.infer_freq(df.index[:500])
    if inferred is None:
        return

    norm = inferred.lower().replace("t", "min")
    if expected_freq not in norm:
        print(
            f"[WARN] Frequência inferida ({inferred}) diferente da esperada ({expected_freq}). "
            "Continuando com o backtest."
        )
