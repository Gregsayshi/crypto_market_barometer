# validation/offline_runner.py  (v1.1)
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

from run_backtest import main as run_backtest_main
from build_validation_inputs import build_inputs

def _resolve_universe_csv(ref: str) -> str:
    p = Path(ref)
    if p.is_dir():
        for name in ("universe_filtered.csv", "universe.csv", "kraken_eur_universe.csv"):
            cand = p / name
            if cand.exists(): return str(cand)
        any_csv = sorted(p.glob("*.csv"))
        if any_csv: return str(any_csv[0])
    return str(p)

def run_offline(cfg_path: Path) -> int:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    meta = cfg.get("meta", {})
    data = cfg.get("data", {})
    a_cfg = cfg.get("model_a", {})
    b_cfg = cfg.get("model_b", {})
    c_cfg = cfg.get("model_c", {})
    allocator_cfg = cfg.get("allocator", {})
    bt = cfg.get("backtest", {})
    out = cfg.get("outputs", {})

    baro_csv = data["barometer_prices"]
    uni_csv  = _resolve_universe_csv(data["universe_prices"])
    prices_wide = _resolve_universe_csv(data.get("prices_wide", uni_csv))
    benches = list(data.get("benchmarks", []))

    # 1) Build exposures + cross-section weights
    inputs_outdir = Path(out.get("inputs_outdir_root", "validation/inputs")) / meta.get("run_tag","run")
    inputs_outdir.mkdir(parents=True, exist_ok=True)
    build_inputs(
        barometer_prices_csv=baro_csv,
        universe_prices_csv=uni_csv,
        a_cfg={"lookbacks": a_cfg.get("lookbacks",[120,180,300]),
               "skip_days": a_cfg.get("skip_days",7),
               "enter_up_z": a_cfg.get("enter_up_z",0.25),
               "exit_up_z":  a_cfg.get("exit_up_z",0.10),
               "enter_down_z": a_cfg.get("enter_down_z",-0.25),
               "exit_down_z":  a_cfg.get("exit_down_z",-0.10),
               "dwell_days": a_cfg.get("dwell_days",5),
               "smooth_ema_days": a_cfg.get("smooth_ema_days",0),
               "symbol_map": data.get("symbol_map", {})},
        b_cfg={"er_window": b_cfg.get("er_window",20),
               "enter_strong": b_cfg.get("enter_strong",0.30),
               "exit_strong":  b_cfg.get("exit_strong",0.25),
               "enter_weak":   b_cfg.get("enter_weak",0.20),
               "exit_weak":    b_cfg.get("exit_weak",0.25),
               "dwell_days":   b_cfg.get("dwell_days",3),
               "smooth_ema_days": b_cfg.get("smooth_ema_days",0)},
        c_cfg={"mode": c_cfg.get("mode","equal_weight"),
               "N": c_cfg.get("N",10),
               "cap": c_cfg.get("cap",0.25),
               "floor_eur": c_cfg.get("floor_eur",250.0),
               "portfolio_eur": c_cfg.get("portfolio_eur",10000.0),
               "skip_days": c_cfg.get("skip_days",3),
               "rebalance": c_cfg.get("rebalance","4W-SUN"),
               "hysteresis": c_cfg.get("hysteresis",3),
               "momo_lb": c_cfg.get("momo_lb",60),
               "start": bt.get("start"),
               "end":   bt.get("end")},
        allocator_cfg={"blend": allocator_cfg.get("blend", {"w_A":0.5,"w_B":0.5}),
                       "exp_caps": allocator_cfg.get("exp_caps", {"uptrend":1.0,"range":1.0,"downtrend":0.0})},
        outdir=str(inputs_outdir)
    )

    exposures = inputs_outdir / "exposures.csv"
    weights   = inputs_outdir / "weights_history.csv"

    # 2) Backtest
    run_dir = Path(bt.get("outdir_root","validation/runs")) / meta.get("run_tag","run")
    argv = [
        "--exposures", str(exposures),
        "--weights",   str(weights),
        "--prices",    str(prices_wide),
        "--signal-lag", str(bt.get("signal_lag",1)),
        "--tc-bps",     str(bt.get("tc_bps",10)),
        "--outdir",     str(run_dir),
    ]
    if bt.get("start"): argv += ["--start", bt["start"]]
    if bt.get("end"):   argv += ["--end",   bt["end"]]
    for b in benches:
        argv += ["--bench", b]

    return run_backtest_main(argv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sys.exit(run_offline(Path(args.config)))

if __name__ == "__main__":
    main()
