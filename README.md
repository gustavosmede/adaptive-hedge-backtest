# BTCUSDT Adaptive Hedge Backtest (Paper-Inspired)

Projeto Python para backtestar uma estratégia de exposição/hedge adaptativo inspirada no artigo **Application of Deep Reinforcement Learning to At-the-Money S&P 500 Options Hedging**, adaptada para **BTCUSDT**.

Este repositório é **educacional e de pesquisa**. Ele não é aconselhamento financeiro, não representa estratégia pronta para produção e não deve ser usado como base única para decisões de investimento.

A implementação prioriza robustez, interpretabilidade e reprodutibilidade com:
- camada de dados Binance com cache e paginação
- features de mercado sem lookahead
- benchmarks (`buy_and_hold`, `buy_and_hold_50pct` e rule-based dinâmico)
- benchmark adicional `buy_and_hold_50pct` para comparação contra alocação parcial estática
- estratégia principal adaptativa determinística (paper-inspired)
- walk-forward out-of-sample
- métricas e gráficos de performance

## Status Atual

O projeto já produz backtests funcionais e comparações úteis entre alocações estáticas e uma política adaptativa. Os experimentos atuais mostram que:
- controle de turnover é decisivo para viabilidade
- a estratégia adaptativa ainda não supera de forma robusta benchmarks passivos simples
- o código é útil para estudo, extensão e experimentação quantitativa

## 1) Estrutura do Projeto

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

### Requisitos
- Python 3.11+

### Instalação

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Crie pastas vazias locais se quiser manter a estrutura explícita:

```bash
mkdir -p data outputs
```

## 3) Execução padrão

Configuração default (igual ao pedido):
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

Rodar:

```bash
python main.py
```

## 4) Exemplo com parâmetros explícitos

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

## 5) O que o pipeline faz

1. Baixa klines da Binance (`spot` ou `futures`) com paginação e rate-limit.
2. Usa cache local em `data/` (parquet/csv) para evitar re-download.
3. Prepara/valida OHLCV.
4. Cria features:
   - retornos 1/3/6/12/24
   - vol realizada curta/longa
   - rolling sharpe
   - média curta/longa e distância
   - z-score do preço
   - momentum
   - ATR percentual
   - volume z-score
   - regime de tendência
   - drawdown de preço
5. Executa backtest com:
   - posição aplicada na barra seguinte (sem lookahead)
   - custo + slippage por mudança de posição
   - equity curve, turnover, custo acumulado, log de trades
6. Avalia walk-forward e concatena resultados OOS.
7. Salva relatórios e gráficos em `outputs/<run_name>/`.

## 6) Resultados e interpretação

Os resultados dependem fortemente de:
- timeframe escolhido
- custo de transação e slippage
- regras de rebalanceamento
- janela de treino e teste do walk-forward

Benchmarks simples como `buy_and_hold` e `buy_and_hold_50pct` são mantidos no projeto justamente para evitar conclusões enganosas sobre ganho de timing.

## 7) Estratégias

- `buy_and_hold`: benchmark 100% exposto.
- `buy_and_hold_50pct`: benchmark estático com 50% de exposição.
- `rule_based_dynamic`: baseline com regras de tendência/volatilidade/drawdown.
- `adaptive_hedge`: estratégia principal inspirada no paper.

### Como a lógica adaptativa replica a ideia central do paper

A estratégia trata hedge/exposição como decisão sequencial: em cada barra ela escolhe um nível de exposição discreto (ex.: `-1, -0.5, 0, 0.5, 1`) maximizando retorno ajustado por risco e custo de transação. O score combina:
- sinal de tendência/momentum
- componente de reversão à média
- penalização por risco (volatilidade e drawdown)
- penalização por churn/custo de troca de posição

Não é DRL nesta versão, mas mantém a estrutura de controle dinâmico de hedge com função objetivo por etapa e fricções de mercado.

## 8) Evitando erros clássicos

- Sem lookahead: decisão em `t`, execução em `t+1`.
- Sem leakage: features só usam janelas passadas.
- Normalização temporal correta: parâmetros de normalização da estratégia adaptativa são ajustados somente na janela de treino de cada fold.
- Walk-forward: treino -> teste subsequente -> avanço temporal.

## 9) Saídas geradas

Em `outputs/<run_name>/`:
- `price.png`
- `equity_comparison.png`
- `drawdown_comparison.png`
- `metrics_summary.csv`
- `<strategy>_bars.csv` (por barra)
- `<strategy>_trades.csv` (rebalance/trades)

Também há resumo final no terminal com tabela de métricas:
- PnL bruto/líquido
- Sharpe, Sortino, Calmar
- CAGR
- Max Drawdown
- Turnover
- Exposição média
- Custo total
- Win rate

## 10) Ajustes rápidos

- Parâmetros globais default: `config.py`
- Overrides por execução: argumentos CLI em `main.py`
- Regras da estratégia adaptativa: `src/strategies/adaptive.py`
- Regras baseline: `src/strategies/baseline.py`

## 11) Limitações

- Estratégia ainda experimental, sem evidência robusta de superioridade fora da amostra contra benchmarks passivos simples.
- Backtest não inclui latência real, filas de execução, impacto de mercado ou custos variáveis.
- A adaptação do paper para BTCUSDT é conceitual; não é reprodução fiel do artigo original.
- O processo de calibração atual é determinístico e relativamente simples.

## 12) Observações

- Dados Binance podem ter gaps eventuais em períodos longos; o validador alerta inconsistências de frequência.
- Para backtest mais rígido, você pode calibrar tamanho de janelas e custos no CLI.
