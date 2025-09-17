#!/usr/bin/env python3
# run_validation.py — one-shot wrapper with timestamped run folders
#
# Behavior change:
# - If you DON'T pass --backtest-outdir, it now creates:
#       validation/runs/YYYYMMDD_HHMMSS(_<run-tag>)
#   so multiple runs never overwrite.
# - If you DON'T pass --outdir for builder (inputs), it creates:
#       validation/inputs/YYYYMMDD_HHMMSS(_<run-tag>)
#
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

def sh(cmd: List[str]) -> int:
    print('>>', ' '.join(map(str, cmd)))
    res = subprocess.run(cmd, check=True)
    return res.returncode

def main(argv=None):
    ap = argparse.ArgumentParser(description='Build historical inputs then run the backtest (one command).')

    # Auto-pick latest outputs dir (Model A/B CSV discovery)
    ap.add_argument('--use-latest-outputs', action='store_true', help='Auto-select latest outputs dir and pick A/B CSVs by pattern.')
    ap.add_argument('--outputs-root', default='model_ensemble/outputs', help='Root folder containing timestamped outputs dirs.')
    ap.add_argument('--a-pattern', default='modelA_output*.csv', help='Glob pattern for Model A CSV inside the chosen outputs dir.')
    ap.add_argument('--b-pattern', default='modelB_output*.csv', help='Glob pattern for Model B CSV inside the chosen outputs dir.')

    # Explicit A/B CSVs
    ap.add_argument('--model-a-csv', help='Model A daily CSV (date index)')
    ap.add_argument('--model-b-csv', help='Model B daily CSV (date index)')

    # Inputs for step 1 (builder)
    ap.add_argument('--kraken-price-csv', required=True, help='Long EUR price CSV (date,asset,close) for weights replay')
    ap.add_argument('--rebalance', default='W-MON', help='Rebalance freq (default: W-MON)')
    ap.add_argument('--N', type=int, default=10, help='Breadth for Model C (default: 10)')
    ap.add_argument('--cap', type=float, default=0.25, help='Per-asset cap (default: 0.25)')
    ap.add_argument('--floor-eur', type=float, default=25.0, help='EUR floor per pos (default: 25)')
    ap.add_argument('--portfolio-eur', type=float, default=1000.0, help='Portfolio notional for floor calc (default: 1000)')
    ap.add_argument('--skip-days', type=int, default=7, help='Skip window for momentum calcs (default: 7)')
    ap.add_argument('--exp-uptrend', type=float, default=1.0, help='Exposure scalar Uptrend')
    ap.add_argument('--exp-range', type=float, default=0.25, help='Exposure scalar Range')
    ap.add_argument('--exp-downtrend', type=float, default=0.0, help='Exposure scalar Downtrend')
    ap.add_argument('--start', type=str, help='Start date YYYY-MM-DD')
    ap.add_argument('--end', type=str, help='End date YYYY-MM-DD')
    ap.add_argument('--outdir', type=str, default='', help='Output folder for exposures.csv & weights_history.csv (defaults to validation/inputs/<ts>)')

    # Inputs for step 2 (backtester)
    ap.add_argument('--prices', required=True, help='Wide prices panel used by the backtester (date, asset columns)')
    ap.add_argument('--bench', action='append', default=[], help='Benchmark asset code(s); pass multiple --bench flags')
    ap.add_argument('--signal-lag', type=int, default=1, help='Signal lag in days (default: 1)')
    ap.add_argument('--tc-bps', type=float, default=10.0, help='Round-trip transaction cost in bps (default: 10)')
    ap.add_argument('--run-tag', default='', help='Optional label appended to run folder name')
    ap.add_argument('--backtest-outdir', default='', help='Optional explicit --outdir for run_backtest.py (overrides run_tag mapping)')
    ap.add_argument('--python', default=sys.executable, help='Python interpreter to use')

    # Paths to the two scripts
    ap.add_argument('--builder-path', default='validation/build_validation_inputs.py', help='Path to build_validation_inputs.py')
    ap.add_argument('--backtest-path', default='validation/run_backtest.py', help='Path to run_backtest.py')

    args = ap.parse_args(argv)

    # Resolve A/B CSVs (manual or latest)
    model_a_csv = args.model_a_csv
    model_b_csv = args.model_b_csv

    if args.use_latest_outputs and (not model_a_csv or not model_b_csv):
        root = Path(args.outputs_root)
        subs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
        latest = max(subs, key=lambda p: p.stat().st_mtime) if subs else None
        if latest is None:
            raise SystemExit(f'No outputs dirs found under {root}')
        def newest(pattern: str):
            matches = sorted(latest.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0] if matches else None
        if not model_a_csv:
            a_path = newest(args.a_pattern)
            if a_path is None: raise SystemExit(f"No A CSV matching '{args.a_pattern}' in {latest}")
            model_a_csv = str(a_path)
        if not model_b_csv:
            b_path = newest(args.b_pattern)
            if b_path is None: raise SystemExit(f"No B CSV matching '{args.b_pattern}' in {latest}")
            model_b_csv = str(b_path)
        print(f'Using latest outputs dir: {latest}')
        print(f'  Model A CSV: {model_a_csv}')
        print(f'  Model B CSV: {model_b_csv}')

    if not model_a_csv or not model_b_csv:
        raise SystemExit('Model A/B CSVs must be provided or discovered via --use-latest-outputs.')

    # Timestamped folders
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_suffix = f'_{args.run_tag}' if args.run_tag else ''

    # Builder outdir (timestamped if not explicitly set)
    inputs_outdir = Path(args.outdir) if args.outdir else Path(f'validation/inputs/{ts}{run_suffix}')
    inputs_outdir.mkdir(parents=True, exist_ok=True)

    # Step 1 — build inputs
    exposures = inputs_outdir / 'exposures.csv'
    weights   = inputs_outdir / 'weights_history.csv'
    cmd1 = [
        args.python, args.builder_path,
        '--model-a-csv', model_a_csv,
        '--model-b-csv', model_b_csv,
        '--kraken-price-csv', args.kraken_price_csv,
        '--rebalance', args.rebalance,
        '--N', str(args.N),
        '--cap', str(args.cap),
        '--floor-eur', str(args.floor_eur),
        '--portfolio-eur', str(args.portfolio_eur),
        '--skip-days', str(args.skip_days),
        '--exp-uptrend', str(args.exp_uptrend),
        '--exp-range', str(args.exp_range),
        '--exp-downtrend', str(args.exp_downtrend),
        '--outdir', str(inputs_outdir),
    ]
    if args.start: cmd1 += ['--start', args.start]
    if args.end:   cmd1 += ['--end', args.end]
    sh(cmd1)

    # Backtester outdir (timestamped if not provided)
    bt_outdir = Path(args.backtest_outdir) if args.backtest_outdir else Path(f'validation/runs/{ts}{run_suffix}')
    bt_outdir.mkdir(parents=True, exist_ok=True)

    # Step 2 — run backtester
    cmd2 = [
        args.python, args.backtest_path,
        '--exposures', str(exposures),
        '--weights', str(weights),
        '--prices', args.prices,
        '--signal-lag', str(args.signal_lag),
        '--tc-bps', str(args.tc_bps),
        '--outdir', str(bt_outdir),
    ]
    if args.start: cmd2 += ['--start', args.start]
    if args.end:   cmd2 += ['--end', args.end]
    for b in (args.bench or []):
        cmd2 += ['--bench', b]

    return sh(cmd2)

if __name__ == '__main__':
    raise SystemExit(main())
