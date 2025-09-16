# Market Barometer

A simple, auditable framework to gauge **direction** and **strength** of crypto market trends and to build a **Kraken EUR** long‑only momentum sleeve. It’s modular: you can run each model alone, run the ensemble, or plug the outputs into the validation/backtest tools.

---

## Components

### Model A — Market direction (TS‑momentum ensemble)
Classifies the barometer (BTC, BTC+ETH 50/50, or BTC+ETH+SOL 1/3) into **Up / Range / Down** using log‑return momentum with a **7‑day skip** and hysteresis + **5‑day dwell**.  
**Outputs:** `close`, z‑scores per lookback, `D`, `state`, `in_dwell`, `reason`.

**Data sources**
- **Live:** CoinGecko daily closes (demo/pro).  
- **Offline (optional):** local long CSV (`date,asset,close`) via `--price-csv` and `--symbol-map`.

**Run**
```bash
# Live (CoinGecko)
python model_a_direction/model_a_v_1.py --barometer-kind btc_eth_sol_333 --days 365 --out data/modelA.csv

# Offline (local CSV written by data/fetch_market_data.py)
python model_a_direction/model_a_v_1.py   --barometer-kind btc_eth_sol_333   --price-csv data/kraken/kraken_daily.csv   --symbol-map "bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR"   --out data/modelA.csv
```

---

### Model B — Trend strength (Efficiency Ratio, ER\_20)
Computes **ER\_20** on log price and classifies **Strong / Neutral / Weak** with hysteresis (0.30/0.25/0.20) and **3‑day dwell**. Combine A+B → **Uptrend / Range / Downtrend** regimes for allocation.  
**Outputs:** `er`, `strength_state`, `in_dwell`, `reason`.

**Run**
```bash
# Live (CoinGecko)
python model_b_strength/model_b_v1.py --barometer-kind btc_eth_sol_333 --days 365 --out data/modelB.csv

# (Model B uses the same offline/local CSV mechanism as Model A if needed.)
```

---

### Model C — Cross‑sectional momentum (Kraken EUR)
Ranks EUR‑quoted Kraken spot pairs by **R90 & R300** (7‑day skip), selects ~**N=10** winners, and emits long‑only **equal weights**, then applies **cap** and **EUR floor**. Excludes stablecoins, leveraged tokens, and dust.  
**Outputs:** `modelC_weights_*.csv` (pair_key, wsname, weight) and `modelC_diagnostics_*.csv` (ranks, scores, R90/R300, ADV30, filters).

**Run (live from Kraken)**
```bash
python model_c_momentum/model_c_v1.py   --max-pairs 120 --since-days 400 --N 10 --cap 0.25   --floor-eur 25 --notional-eur 1000   --out-weights model_ensemble/outputs/modelC_weights.csv   --out-diag    model_ensemble/outputs/modelC_diagnostics.csv
```

**Run (offline from a prebuilt CSV of Kraken OHLC for Model C)**  
Use `data/kraken_list_eur_pairs.py` to list EUR pairs, and `data/fetch_market_data.py` to build a long price CSV if desired.
```bash
python model_c_momentum/model_c_v1.py   --price-csv data/kraken/kraken_daily.csv   --csv-key altname \  # or pair_key / wsname, matching the CSV’s asset naming
  --since-days 400 --N 10 --cap 0.25 --floor-eur 25 --notional-eur 1000
```

---

## Ensemble Orchestration

`model_ensemble/run_ensemble.py` runs A, B, C and scales Model C weights by an **exposure scalar** determined from A+B’s regime. Outputs land in `model_ensemble/outputs/<UTC_TS>/`.

**Quick start (YAML‑driven)**
```bash
python model_ensemble/run_ensemble.py --config config.yaml
```

**Example `config.yaml`**
```yaml
meta:
  output_root: model_ensemble/outputs

barometer:
  kind: btc_eth_sol_333  # bitcoin|ethereum|solana|btc_eth_5050|btc_eth_sol_333
  # Optional: use a local CSV for A/B instead of CoinGecko
  # price_csv: data/kraken/kraken_daily.csv
  # symbol_map: "bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR"

data:
  coingecko:
    api_tier: demo      # demo|pro
    vs_currency: usd
    days: 365
    api_key_file: null  # e.g., ./coingecko.key

  kraken:
    max_pairs: 120
    since_days: 400
    # Optional offline for Model C:
    # price_csv: data/kraken/kraken_daily.csv
    # csv_key: altname   # or pair_key|wsname to match your CSV

model_c:
  N_target: 10
  N_min: 8
  N_max: 12
  weight_cap: 0.25
  notional_floor_eur: 25.0
  portfolio_notional_eur: 1000.0

allocator:
  exposure_scalars:
    uptrend: 1.0
    range:   0.25
    downtrend: 0.0
```

**CLI overrides (examples)**
```bash
python model_ensemble/run_ensemble.py   --config config.yaml   --barometer-kind btc_eth_5050   --days-ab 365   --N 8 --since-days-c 450   --exp-range 0.33
```

**Outputs**
- `modelA_output_*.csv`, `modelB_output_*.csv`  
- `modelC_weights_*.csv`, `modelC_diagnostics_*.csv`  
- `final_allocation_<Regime>_*.csv` (Model C weights × exposure scalar)  
- `run_summary_*.json` (A/B latest states, regime, scalar, weight sums)

---

## Validation & Backtesting (minimal)

A light “drop‑in” backtester lives in `validation/`. It consumes:
- **Exposures**: CSV (`date,exposure_scalar,regime`) or `run_summary_*.json`
- **Weights**: CSV (`date,asset,weight`) — daily or rebalance rows
- **Prices**: long CSV (`date,asset,close`) — daily EUR

**Example**
```bash
# Build prices from latest ensemble’s selected assets (+ optional baselines)
python validation/build_prices_kraken.py   --from-ensemble-latest --since-days 400   --asset-col pair_key --include XBTEUR   --out data/prices_eur.csv

# Backtest strategy vs baselines
python validation/run_backtest.py   --from-ensemble-latest   --prices data/prices_eur.csv   --bench XBTEUR --bench ETHEUR
```

Outputs: `metrics_summary.csv` and charts (equity, drawdowns, rolling vol/sharpe, monthly heatmap, regime ribbon, turnover, weights stack).

---

## Data Utilities

- `data/fetch_market_data.py` — grab Binance (`/api/v3/klines`) and/or Kraken OHLC into long CSVs.  
- `data/kraken_list_eur_pairs.py` — list all EUR‑quoted Kraken pairs for Model C universe curation.

**Examples**
```bash
# Binance: 1000 daily candles since 2022-01-01 (USDT quotes)
python data/fetch_market_data.py binance   --symbols BTCUSDT ETHUSDT SOLUSDT   --interval 1d --limit 1000 --since 2022-01-01   --out data/binance/spot_daily.csv

# Kraken: EUR daily closes since 2022-01-01
python data/fetch_market_data.py kraken   --symbols XBTEUR ETHEUR SOLEUR   --since 2022-01-01   --out data/kraken/kraken_daily.csv
```

---

## Dev & Testing

- Python **3.10+**  
- Install: `pip install -r requirements.txt` (pandas, numpy, requests, matplotlib, pyyaml, pytest)
- Run tests: `python -m pytest -q`

---

## Troubleshooting

- **“Weights date after last price date”** → rebuild prices with a later `--end` or include more days.  
- **No assets selected in Model C** → raise `--max-pairs`, check dust/filters, or extend `--since-days`.  
- **Offline CSVs** → ensure columns are exactly `date,asset,close` and your `--symbol-map`/`--csv-key` matches the CSV’s asset naming.

---
