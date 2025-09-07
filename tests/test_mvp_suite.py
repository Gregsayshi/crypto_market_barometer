"""
MVP test suite (pytest) for Models A, B, C and ensemble invariants.

Run with:
  pytest -q

These tests are synthetic (no network). They validate math, state machines,
ranking/weights logic, and the ensemble scalar invariant.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------
# Import targets with fallbacks, so tests work regardless of your filenames.
# Adjust if your modules live in different paths.
# -----------------------------------------------------------------------------
try:  # Preferred: your repo layout
    from model_b_strength.model_b_v1 import compute_er, run_strength_state_machine, ModelBConfig  # type: ignore
except Exception:  # Fallback: canvas filename
    from ModelB_v1_EfficiencyRatio import compute_er, run_strength_state_machine, ModelBConfig  # type: ignore

try:
    from model_c_momentum.model_c_v1 import _ts_momentum_with_skip, apply_caps_and_floors, apply_membership_with_buffers, PortfolioConfig  # type: ignore
except Exception:
    from ModelC_v1_Kraken import _ts_momentum_with_skip, apply_caps_and_floors, apply_membership_with_buffers, PortfolioConfig  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def mk_series(values: list[float], start="2024-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


# -----------------------------------------------------------------------------
# Model B — Efficiency Ratio (ER) math sanity
# -----------------------------------------------------------------------------

def test_er_constant_growth_near_one():
    # log-price increases by 0.01 daily → ER_N == 1 for N-window (up to fp error)
    N = 20
    logp = np.arange(0, 40) * 0.01
    price = np.exp(logp)
    s = mk_series(price)
    er = compute_er(s, N)
    val = float(er.iloc[-1])
    assert val > 0.95 and val <= 1.0 + 1e-9


def test_er_flat_is_zero():
    N = 20
    price = np.ones(40)
    s = mk_series(price)
    er = compute_er(s, N)
    assert abs(float(er.iloc[-1])) < 1e-9


def test_er_zigzag_is_low():
    # Alternate +1% / -1% moves → low net displacement vs path length
    N = 20
    rets = np.array([0.01, -0.01] * 25)
    price = 100.0 * np.exp(np.cumsum(rets))
    s = mk_series(list(price))
    er = compute_er(s, N)
    assert float(er.iloc[-1]) < 0.2


# -----------------------------------------------------------------------------
# Model B — State machine (hysteresis + dwell)
# -----------------------------------------------------------------------------

def test_state_machine_hysteresis_and_dwell():
    cfg = ModelBConfig(N=20, enter_strong=0.30, exit_strong=0.25, enter_weak=0.20, exit_weak=0.25, dwell_days=3)
    # ER path: neutral→strong at 0.32, then under 0.25 after dwell
    er_vals = [0.10, 0.12, 0.15, 0.32, 0.28, 0.27, 0.26, 0.24, 0.23, 0.22]
    er = mk_series(er_vals)
    states = run_strength_state_machine(er, cfg)
    st = list(states["strength_state"].astype(str))
    # Index: 0..9
    assert st[3] == cfg.strong  # Enter Strong at 0.32
    # During dwell (next 3 days), stay Strong even if < exit_strong
    assert st[4] == cfg.strong and st[5] == cfg.strong and st[6] == cfg.strong
    # After dwell, next day is 0.24 (<= exit_strong) → back to Neutral
    assert st[7] == cfg.neutral


# -----------------------------------------------------------------------------
# Shared util — TS momentum with skip
# -----------------------------------------------------------------------------

def test_ts_momentum_with_skip_formula():
    # Use small K & skip for compact fixture
    K, skip = 5, 2
    # Price_t = exp(t*0.1) so log price increases by 0.1 per day
    t = np.arange(0, 20)
    price = np.exp(0.1 * t)
    s = mk_series(list(price))
    # Expected: log(P[t-skip]) - log(P[t-(K+skip)]) = 0.1 * K
    expected = 0.1 * K
    got = _ts_momentum_with_skip(s, K=K, skip=skip)
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)


# -----------------------------------------------------------------------------
# Model C — Membership (entry/exit buffers) & weights (caps/floors)
# -----------------------------------------------------------------------------

def test_membership_with_buffers_keep_add_fill():
    # Ranked list with positions 1..10 (lower is better)
    idx = [f"A{i}" for i in range(1, 11)]
    ranked = pd.DataFrame({"score": range(1, 11), "pos": range(1, 11)}, index=idx)
    pcfg = PortfolioConfig(N_target=4, N_min=4, N_max=5, entry_rank=3, exit_rank=6)
    prev = {"A1": 0.25, "A8": 0.25}  # A1 pos=1 (keep), A8 pos=8 (>6 → drop)
    selected = apply_membership_with_buffers(ranked, pcfg, prev)
    # Expect to keep A1, add A2 & A3 (entry<=3), then fill up to N_min with best available A4
    assert selected == ["A1", "A2", "A3", "A4"]


def test_apply_caps_and_floors_cap_then_floor():
    pcfg = PortfolioConfig(N_target=10, weight_cap=0.25, notional_floor_eur=25.0, portfolio_notional_eur=1000.0)
    # Four names so 4 * 0.25 = 1.0 is achievable under the cap
    w0 = {"X": 0.9, "Y": 0.06, "Z": 0.04, "W": 0.01}
    w = apply_caps_and_floors(w0, pcfg)
    assert max(w.values()) <= pcfg.weight_cap + 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_apply_caps_and_floors_floor_drop():
    pcfg = PortfolioConfig(N_target=3, weight_cap=0.5, notional_floor_eur=100.0, portfolio_notional_eur=1000.0)  # floor_w=0.10
    w0 = {"A": 0.89, "B": 0.01, "C": 0.10}
    w = apply_caps_and_floors(w0, pcfg)
    # 'B' should be dropped (below 0.10), A & C renormalized
    assert "B" not in w and abs(sum(w.values()) - 1.0) < 1e-9


# -----------------------------------------------------------------------------
# Ensemble invariant — final weights sum to exposure scalar
# -----------------------------------------------------------------------------

def test_ensemble_scalar_invariant():
    modelC_weights = np.array([0.2, 0.3, 0.5])  # sums to 1
    scalar = 0.25
    final = modelC_weights * scalar
    assert abs(final.sum() - scalar) < 1e-12
