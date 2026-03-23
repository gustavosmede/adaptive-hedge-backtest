from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG, ProjectConfig
from src.backtest.engine import BacktestParams, run_backtest
from src.backtest.walk_forward import WalkForwardParams, run_walk_forward
from src.data.binance_data import download_binance_klines, prepare_ohlcv, validate_data
from src.evaluation.metrics import compute_metrics, summarize_metrics
from src.evaluation.reports import (
    plot_drawdown_comparison,
    plot_equity_comparison,
    plot_price,
    save_metrics_table,
    save_strategy_outputs,
)
from src.features.engineer import build_features
from src.strategies.adaptive import AdaptiveHedgeStrategy
from src.strategies.baseline import BuyAndHoldStrategy, RuleBasedDynamicStrategy
from src.utils.io import ensure_dir, safe_run_name
from src.utils.timeframe import annualization_factor


def parse_args(cfg: ProjectConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest de hedge/exposição adaptativa BTCUSDT")
    parser.add_argument("--symbol", default=cfg.data.symbol)
    parser.add_argument("--market", choices=["spot", "futures"], default=cfg.data.market)
    parser.add_argument("--interval", choices=["5m", "15m", "30m", "1h"], default=cfg.data.interval)
    parser.add_argument("--start-date", default=cfg.data.start_date)
    parser.add_argument("--end-date", default=cfg.data.end_date)
    parser.add_argument("--transaction-cost", type=float, default=cfg.backtest.transaction_cost)
    parser.add_argument("--slippage", type=float, default=cfg.backtest.slippage)
    parser.add_argument("--initial-capital", type=float, default=cfg.backtest.initial_capital)
    parser.set_defaults(allow_short=cfg.backtest.allow_short)
    parser.add_argument("--allow-short", dest="allow_short", action="store_true", help="Permite exposição negativa")
    parser.add_argument("--long-only", dest="allow_short", action="store_false", help="Restringe exposição entre 0 e 1")
    parser.add_argument("--min-holding-bars", type=int, default=cfg.backtest.min_holding_bars)
    parser.add_argument("--rebalance-every-n-bars", type=int, default=cfg.backtest.rebalance_every_n_bars)
    parser.add_argument("--max-position-step", type=float, default=cfg.backtest.max_position_step)
    parser.add_argument("--disable-walk-forward", action="store_true")
    parser.add_argument("--train-bars", type=int, default=cfg.walk_forward.train_bars)
    parser.add_argument("--test-bars", type=int, default=cfg.walk_forward.test_bars)
    parser.add_argument("--step-bars", type=int, default=cfg.walk_forward.step_bars)
    parser.add_argument("--run-name", default=cfg.output.run_name)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace, cfg: ProjectConfig) -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = ensure_dir(project_root / cfg.data.data_dir)
    outputs_root = ensure_dir(project_root / cfg.output.outputs_dir)
    run_dir = ensure_dir(outputs_root / safe_run_name(args.run_name))

    print("[1/6] Baixando/carregando dados da Binance...")
    raw = download_binance_klines(
        symbol=args.symbol,
        market=args.market,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=data_dir,
        cache_format=cfg.data.cache_format,
        use_cache=cfg.data.use_cache,
        sleep_seconds=cfg.data.sleep_seconds,
    )

    print("[2/6] Preparando OHLCV e validando dataset...")
    ohlcv = prepare_ohlcv(raw)
    validate_data(ohlcv, interval=args.interval)

    print("[3/6] Construindo features sem lookahead...")
    df = build_features(ohlcv)

    bt_params = BacktestParams(
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        initial_capital=args.initial_capital,
        min_holding_bars=args.min_holding_bars,
        rebalance_every_n_bars=args.rebalance_every_n_bars,
        max_position_step=args.max_position_step,
    )

    strategies = {
        "buy_and_hold": BuyAndHoldStrategy(target=1.0),
        "buy_and_hold_50pct": BuyAndHoldStrategy(target=0.5),
        "rule_based_dynamic": RuleBasedDynamicStrategy(allow_short=args.allow_short),
        "adaptive_hedge": AdaptiveHedgeStrategy(
            cfg=cfg.strategy,
            allow_short=args.allow_short,
            execution_cost=args.transaction_cost + args.slippage,
        ),
    }

    print("[4/6] Executando backtest...")
    if not args.disable_walk_forward:
        wf_params = WalkForwardParams(
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
        )
        results = run_walk_forward(df, strategies, bt_params, wf_params)
    else:
        results = {}
        for name, strategy in strategies.items():
            strategy.fit(df)
            bt = run_backtest(df, strategy, bt_params)
            results[name] = type("S", (), {"bars": bt.bars, "trades": bt.trades})

    print("[5/6] Calculando métricas e salvando saídas...")
    ann = annualization_factor(args.interval)
    metrics_map: dict[str, dict[str, float]] = {}
    bars_map = {}

    for name, result in results.items():
        bars = result.bars.copy()
        trades = result.trades.copy()
        bars_map[name] = bars
        metrics_map[name] = compute_metrics(bars, ann)
        save_strategy_outputs(name, bars, trades, run_dir)

    metrics_df = summarize_metrics(metrics_map)
    save_metrics_table(metrics_df, run_dir / "metrics_summary.csv")

    print("[6/6] Gerando gráficos...")
    plot_price(df, run_dir / "price.png")
    plot_equity_comparison(bars_map, run_dir / "equity_comparison.png")
    plot_drawdown_comparison(bars_map, run_dir / "drawdown_comparison.png")

    print("\n=== RESUMO FINAL ===")
    print(metrics_df.round(6).to_string())
    print(f"\nSaídas em: {run_dir}")


def main() -> None:
    cfg = DEFAULT_CONFIG
    args = parse_args(cfg)
    run_pipeline(args, cfg)


if __name__ == "__main__":
    main()
