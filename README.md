## Market Barometer

This module aims to provide a simple, yet good enough gauge on the direction and strength of the current price trend in cryptocurrency markets. Will be using Kraken for the selection of a model portfolio.

### Model A: Market direction
Model A is a price-only market direction gauge that classifies the crypto barometer (BTC or 50/50 BTC+ETH or 33/33/33 BTC+ETH+SOL) as Up / Range / Down. It uses a time-series momentum ensemble (lookbacks 120/180/300 with a 7-day skip), applies hysteresis and a 5-day dwell, and emits both a discrete state and a continuous strength score D—meant to scale exposure, not to be a standalone trading strategy. Data come from CoinGecko daily closes (demo-tier API key).

### Model B: Trend strength
Model B is a trend-strength gauge based on the Efficiency Ratio (ER_20) computed on log prices. It outputs a continuous score and a discrete strength_state ∈ {Strong, Neutral, Weak}, with hysteresis (0.30/0.25/0.20 bands) and a 3-day dwell to reduce flip-flops. Input is the same barometer as Model A (BTC, BTC+ETH 50/50, or BTC+ETH+SOL 1/3) from CoinGecko daily closes. The CSV includes er, strength_state, in_dwell, and reason; it’s intended to be combined with Model A’s direction to form Uptrend / Range / Downtrend regimes for the allocator.

### Model C: Cross-sectional Momentum
Model C is a cross-sectional momentum portfolio that works over Kraken’s EUR-quoted spot universe. Each rebalance (weekly/biweekly), it ranks assets by R90 and R300 log-return momentum with a 7-day skip, selects ~10 winners, and outputs long-only weights (equal-weight by default) after applying a per-name cap and a floor. Stablecoins, leveraged tokens, and “dust” names are excluded; diagnostics include ranks, scores, and basic liquidity stats for auditability. Model C is alpha-only—gross exposure is set later by Models A (direction) and B (trend strength) in the allocator.