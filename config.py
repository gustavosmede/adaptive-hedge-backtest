from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    market: str = "spot"  # spot | futures
    interval: str = "15m"  # 5m | 15m | 30m | 1h
    start_date: str = "2021-01-01"
    end_date: str = "2025-01-01"
    data_dir: Path = Path("data")
    cache_format: str = "parquet"  # parquet | csv
    use_cache: bool = True
    sleep_seconds: float = 0.15


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    transaction_cost: float = 0.0004
    slippage: float = 0.0002
    allow_short: bool = False
    min_holding_bars: int = 8
    rebalance_every_n_bars: int = 4
    max_position_step: float = 0.25


@dataclass(slots=True)
class WalkForwardConfig:
    enabled: bool = True
    train_bars: int = 20_000
    test_bars: int = 5_000
    step_bars: int = 5_000


@dataclass(slots=True)
class StrategyConfig:
    rebalance_levels_long_only: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    rebalance_levels_long_short: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    risk_aversion: float = 0.55
    cost_aversion: float = 0.90
    trend_weight: float = 0.70
    momentum_weight: float = 0.60
    mean_reversion_weight: float = 0.30
    vol_penalty_weight: float = 0.65
    drawdown_penalty_weight: float = 0.85
    hysteresis_band: float = 0.30
    vol_target: float = 0.018


@dataclass(slots=True)
class OutputConfig:
    outputs_dir: Path = Path("outputs")
    run_name: str = "btc_adaptive_backtest"


@dataclass(slots=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_CONFIG = ProjectConfig()
