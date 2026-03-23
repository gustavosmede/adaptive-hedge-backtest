# BTCUSDT Adaptive Hedge Backtest (Paper-Inspired)

Python project to backtest an adaptive exposure/hedge strategy inspired by the paper **Application of Deep Reinforcement Learning to At-the-Money S&P 500 Options Hedging**, adapted for **BTCUSDT**.

This repository is **educational and for research**. It is not financial advice, does not represent a production-ready strategy, and should not be used as the sole basis for investment decisions.

The implementation prioritizes robustness, interpretability, and reproducibility with:
- Binance data layer with caching and pagination
- Market features without lookahead
- Benchmarks (`buy_and_hold`, `buy_and_hold_50pct`, and dynamic rule-based)
- Additional `buy_and_hold_50pct` benchmark for comparison against static partial allocation
- Deterministic adaptive main strategy (paper-inspired)
- Walk-forward out-of-sample
- Performance metrics and charts

## Current Status

The project already produces functional backtests and useful comparisons between static allocations and an adaptive policy. Current experiments show that:
- Turnover control is decisive for viability
- The adaptive strategy does not yet robustly outperform simple passive benchmarks
- The code is useful for study, extension, and quantitative experimentation

## 1) Project Structure

```text
project/
  data/
  outputs/
  src/
    data/
      __init__.py
      binance_data.py
    features/
      __init__.py
      engineer.py
    strategies/
      __init__.py
      base.py
      baseline.py
      adaptive.py
    backtest/
      __init__.py
      engine.py
      walk_forward.py
    evaluation/
      __init__.py
      metrics.py
      reports.py
    utils/
      __init__.py
      io.py
      timeframe.py
    __init__.py
  config.py
  main.py
  requirements.txt
  README.md
```

## 2) Setup

### Requirements
- Python 3.11+

### Installation

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create empty local folders if you want to maintain the explicit structure:

```bash
mkdir -p data outputs
```

## 3) Default Execution

Default configuration (as requested):
- `symbol=BTCUSDT`
- `market=spot`
- `interval=15m`
- `start_date=2021-01-01`
- `end_date=2025-01-01`
- `transaction_cost=0.0004`
- `slippage=0.0002`
- `initial_capital=10000`
- `long_only=True`
- `min_holding_bars=8`
- `rebalance_every_n_bars=4`
- `max_position_step=0.25`

Run:

```bash
python main.py
```

## 4) Example with explicit parameters

```bash
python main.py \
  --symbol BTCUSDT \
  --market spot \
  --interval 15m \
  --start-date 2021-01-01 \
  --end-date 2025-01-01 \
  --transaction-cost 0.0004 \
  --slippage 0.0002 \
  --initial-capital 10000 \
  --long-only \
  --min-holding-bars 8 \
  --rebalance-every-n-bars 4 \
  --max-position-step 0.25 \
  --train-bars 20000 \
  --test-bars 5000 \
  --step-bars 5000 \
  --run-name btc_wf_default
```

## 5) What the pipeline does

1. Downloads klines from Binance (`spot` or `futures`) with pagination and rate-limiting.
2. Uses local cache in `data/` (parquet/csv) to avoid re-downloading.
3. Prepares/validates OHLCV.
4. Creates features:
   - 1/3/6/12/24 returns
   - short/long realized vol
   - rolling sharpe
   - short/long average and distance
   - price z-score
   - momentum
   - percentage ATR
   - volume z-score
   - trend regime
   - price drawdown
5. Executes backtest with:
   - position applied on the next bar (no lookahead)
   - cost + slippage per position change
   - equity curve, turnover, cumulative cost, trade log
6. Evaluates walk-forward and concatenates OOS results.
7. Saves reports and charts in `outputs/<run_name>/`.

## 6) Results and interpretation

Results depend heavily on:
- chosen timeframe
- transaction cost and slippage
- rebalancing rules
- walk-forward training and testing windows

Simple benchmarks like `buy_and_hold` and `buy_and_hold_50pct` are kept in the project specifically to avoid misleading conclusions about timing gains.

## 7) Strategies

- `buy_and_hold`: 100% exposed benchmark.
- `buy_and_hold_50pct`: static benchmark with 50% exposure.
- `rule_based_dynamic`: baseline with trend/volatility/drawdown rules.
- `adaptive_hedge`: main strategy inspired by the paper.

### How the adaptive logic replicates the paper's core idea

The strategy treats hedge/exposure as a sequential decision: at each bar, it chooses a discrete exposure level (e.g., `-1, -0.5, 0, 0.5, 1`) maximizing risk-adjusted return and transaction cost. The score combines:
- trend/momentum signal
- mean reversion component
- risk penalty (volatility and drawdown)
- churn/position change cost penalty

It is not DRL in this version, but it maintains the dynamic hedge control structure with a per-step objective function and market frictions.

## 8) Avoiding classic errors

- No lookahead: decision at `t`, execution at `t+1`.
- No leakage: features only use past windows.
- Correct temporal normalization: normalization parameters for the adaptive strategy are fitted only on the training window of each fold.
- Walk-forward: train -> subsequent test -> temporal advance.

## 9) Generated outputs

In `outputs/<run_name>/`:
- `price.png`
- `equity_comparison.png`
- `drawdown_comparison.png`
- `metrics_summary.csv`
- `<strategy>_bars.csv` (per bar)
- `<strategy>_trades.csv` (rebalance/trades)

There is also a final summary in the terminal with a metrics table:
- Gross/net PnL
- Sharpe, Sortino, Calmar
- CAGR
- Max Drawdown
- Turnover
- Average exposure
- Total cost
- Win rate

## 10) Quick adjustments

- Default global parameters: `config.py`
- Per-execution overrides: CLI arguments in `main.py`
- Adaptive strategy rules: `src/strategies/adaptive.py`
- Baseline rules: `src/strategies/baseline.py`

## 11) Limitations

- Strategy is still experimental, without robust evidence of out-of-sample superiority against simple passive benchmarks.
- Backtest does not include real latency, execution queues, market impact, or variable costs.
- The adaptation of the paper for BTCUSDT is conceptual; it is not a faithful reproduction of the original article.
- The current calibration process is deterministic and relatively simple.

## 12) Observations

- Binance data may have occasional gaps over long periods; the validator warns of frequency inconsistencies.
- For a stricter backtest, you can calibrate window sizes and costs in the CLI.
