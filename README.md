# Crypto Market Barometer v1.1 — Liquidity-Filtered Macro Allocator

A modular, **offline-first** research framework for evaluating crypto market direction and allocating capital within a **liquidity-filtered Kraken EUR universe**.

Version 1.1 refactors the original three-model ensemble into a **macro allocator**:

- **Model A** – Market direction  
- **Model B** – Market strength  
- **Allocator** – blends A + B into a dynamic exposure scalar (cash ↔ tokens)  
- **Model C** – light cross-sectional allocator (equal or tilted weights) over a pre-filtered liquid universe  

All runs are deterministic, timestamped, and reproducible.

---

## ⚙️ Architecture Overview

```
config.yaml
   │
   ▼
validation/offline_runner.py
   ├─▶ build_validation_inputs.py   → exposures.csv, weights_history.csv
   ├─▶ run_backtest.py              → full performance evaluation
   └─▶ outputs under validation/{inputs,runs}/<run_tag>_<UTC_TS>/
```

### Core Principles

- **Offline data** – all inputs are CSV-based; no API calls required.  
- **Point-in-time universe** – tradable pairs filtered by liquidity, age, and quality.  
- **Transparent allocator** – exposure timing (A + B) and asset selection (C) are cleanly separated.  
- **Deterministic validation** – every run is logged under a timestamped directory.

---

## 🧩 Components

### Model A — Direction

Computes a directional signal from barometer assets (BTC, ETH, SOL) using multi-lookback momentum, skip, hysteresis, and dwell logic.  
**Output:** normalized scalar in `[0, 1]`.

### Model B — Strength

Measures trend breadth/efficiency using an Efficiency Ratio (ER).  
**Output:** normalized scalar in `[0, 1]`.

### Allocator (A + B)

Blends Model A and Model B into an **exposure scalar**, then applies regime caps:

```yaml
allocator:
  blend: { w_A: 0.5, w_B: 0.5 }
  exp_caps: { uptrend: 1.0, range: 1.0, downtrend: 0.0 }
```

### Model C — Cross-Sectional Allocator

Operates only on pre-filtered liquid pairs. Modes supported:

- `equal_weight` (default, stable and low turnover)  
- `light_momo` (gentle momentum tilt)  
- `cap_weight` or `vol_target` (optional schemes)

**Output:** normalized long-only weights for eligible assets.

---

## 🧮 Universe Building

Build tradable assets by **liquidity and data-quality filters**:

```bash
python data/build_tradeable_universe.py   --in data/kraken/kraken_eur_universe.csv   --out-dir data/universe_subset/universe_v1   --window-days 365   --min-age-days 1   --min-coverage 0.95   --min-adv30-eur 200000   --min-nonzero-vol 0.90   --max-close-vwap-dev 0.05   --max-zero-return 0.20   --vol-min 1 --vol-max 10   --whitelist XBTEUR ETHEUR SOLEUR   --exclude-pattern "^ZUSD|^USDT|.*3L$|.*3S$|^W[A-Z]+EUR"
```

**Outputs**

- `universe_filtered.csv` – long-format prices (date, asset, close, vwap, volume)  
- `eligible_assets.csv` – final list of tradeable pairs  
- `manifest.json` – recorded thresholds and parameters  

Use `universe_filtered.csv` as the canonical input for backtests.

---

## 🧰 Validation & Backtesting

Run a full validation (build inputs + backtest) in one command:

```bash
python validation/offline_runner.py --config config.yaml
```

Each run automatically creates timestamped folders:

```
validation/inputs/<run_tag>_<UTC_TS>/
validation/runs/<run_tag>_<UTC_TS>/
```

### Example `config.yaml`

```yaml
meta:
  run_tag: prod_like
  timezone: UTC

data:
  barometer_prices: data/kraken/kraken_daily.csv
  universe_prices:  data/universe_subset/universe_v7/universe_filtered.csv
  prices_wide:      data/universe_subset/universe_v7/universe_filtered.csv
  benchmarks: [XBTEUR, ETHEUR]

  symbol_map:
    bitcoin:  XBTEUR
    ethereum: ETHEUR
    solana:   SOLEUR

model_a:
  barometer_kind: btc
  lookbacks: [120, 180, 300]
  skip_days: 7
  enter_up_z: 0.25
  exit_up_z: 0.10
  enter_down_z: -0.25
  exit_down_z: -0.10
  dwell_days: 5

model_b:
  barometer_kind: btc_eth_sol_333
  er_window: 20
  enter_strong: 0.30
  exit_strong: 0.25
  enter_weak: 0.20
  exit_weak: 0.25
  dwell_days: 3

model_c:
  mode: equal_weight
  rebalance: 4W-SUN
  N: 10
  cap: 0.25
  floor_eur: 250
  portfolio_eur: 10000
  skip_days: 3

allocator:
  blend: { w_A: 0.5, w_B: 0.5 }
  exp_caps: { uptrend: 1.0, range: 1.0, downtrend: 0.0 }

backtest:
  start: "2024-09-19"
  end:   "2025-09-19"
  signal_lag: 1
  tc_bps: 10
  outdir_root: validation/runs

outputs:
  write_inputs: true
  inputs_outdir_root: validation/inputs
```

### Results

**Produced files**

- `metrics_summary.csv` – CAGR, vol, Sharpe, max DD, turnover  
- `equity_curves.png`, `drawdowns.png`, `rolling_sharpe.png`, `rolling_vol.png`,  
- `weights_stack.png`, `regime_ribbon.png`, `turnover.png`  

---

## 📦 Data Utilities

| Script | Purpose |
|-------:|:--------|
| `data/fetch_market_data.py` | Fetch OHLC data from Binance or Kraken into long CSVs |
| `data/build_tradeable_universe.py` | Create filtered tradable universe based on liquidity & quality |
| `data/build_kraken_universe.py` | Optional: rebuild raw Kraken pair history for local use |

**Examples**

```bash
# Kraken EUR daily closes
python data/fetch_market_data.py kraken   --symbols XBTEUR ETHEUR SOLEUR   --since 2022-01-01   --out data/kraken/kraken_daily.csv
```


## 🧭 Repository Layout

```
crypto_market_barometer/
│
├── data/                     # Raw + derived CSV data utilities
├── model_a_direction/        # Directional model
├── model_b_strength/         # Strength model
├── model_c_momentum/         # Cross-sectional allocator
├── validation/               # Offline runner, builders, backtester
└── universe_subset/          # Point-in-time filtered universes
```

