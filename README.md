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

The `validation/` folder now supports **timestamped, reproducible backtests** using a single wrapper.

### One-shot validation run
Use `validation/run_validation.py` to build exposures & weights history **and** immediately run the backtest:

```bash
python validation/run_validation.py \
  --use-latest-outputs \
  --kraken-price-csv data/kraken/kraken_eur_universe.csv \
  --prices data/prices_eur.csv \
  --bench XBTEUR --bench ETHEUR \
  --start 2024-09-17 --end 2025-09-16 \
  --rebalance W-MON --N 10 --cap 0.25 --floor-eur 25 --portfolio-eur 1000 \
  --signal-lag 1 --tc-bps 10 \
  --run-tag barometer_v1
```

**What it does:**
1. Picks the latest `model_ensemble/outputs/<timestamp>/` folder (or use `--model-a-csv` / `--model-b-csv` to override)
2. Builds:
   - `exposures.csv` (daily regime + exposure scalar)
   - `weights_history.csv` (walk-forward Model C weights)
3. Runs `validation/run_backtest.py` with those inputs
4. Writes results to:
   - Inputs: `validation/inputs/YYYYMMDD_HHMMSS(_<tag>)/`
   - Results: `validation/runs/YYYYMMDD_HHMMSS(_<tag>)/`

You can re-run many times — timestamped folders avoid overwriting old results.

### YAML-first workflow
Instead of CLI flags, store parameters in `validation_config.yaml`:

```yaml
builder:
  use_latest_outputs: true
  outputs_root: model_ensemble/outputs
  kraken_price_csv: data/kraken/kraken_eur_universe.csv
  rebalance: W-MON
  N: 10
  cap: 0.25
  floor_eur: 25
  portfolio_eur: 1000
  exp_uptrend: 1.0
  exp_range: 0.25
  exp_downtrend: 0.0
  start: 2024-09-17
  end: 2025-09-16

backtester:
  prices: data/prices_eur.csv
  benches: [XBTEUR, ETHEUR]
  signal_lag: 1
  tc_bps: 10
  run_tag: barometer_v1

paths:
  builder_path: validation/build_validation_inputs.py
  backtest_path: validation/run_backtest.py
```

Run with:
```bash
python validation/run_validation_from_yaml.py --config validation_config.yaml
```

This produces the same timestamped results and is ideal for reproducible runs in CI/CD or notebooks.

### Outputs
- **metrics_summary.csv** – CAGR, vol, Sharpe, max DD, turnover
- **Charts** – equity curve, drawdowns, rolling Sharpe/vol, monthly heatmap, regime ribbon, weights stack

You can now safely compare multiple runs side-by-side by browsing `validation/runs/` and diffing the `metrics_summary.csv` files.

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
