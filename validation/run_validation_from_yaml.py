#!/usr/bin/env python3
# run_validation_from_yaml.py — YAML-first launcher (no external deps)
#
# Reads a *simple* YAML (our template) and invokes run_validation.py with the right flags.
# Supports scalars and simple lists; expects sections: builder, backtester, paths.
#
import argparse
import sys
import subprocess
from pathlib import Path

def parse_simple_yaml(path):
    # Very small YAML subset parser tailored to our template
    # Supports: key: value, key: [a,b,c], nested sections with 2-space indents
    data = {}
    stack = [data]
    indent_levels = [0]
    last_key_stack = [None]

    def coerce(val):
        if val in ("null", "None", "~"): return None
        if val in ("true", "True"): return True
        if val in ("false", "False"): return False
        try:
            if val.isdigit(): return int(val)
            if val.replace('.','',1).isdigit() and val.count('.') < 2: return float(val)
        except Exception:
            pass
        return val

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#',1)[0].rstrip()
            if not line: continue
            indent = len(line) - len(line.lstrip(' '))
            if indent % 2 != 0:
                raise SystemExit(f"Invalid indent: use multiples of 2 spaces -> '{raw.rstrip()}'")
            while indent < indent_levels[-1]:
                stack.pop(); indent_levels.pop(); last_key_stack.pop()
            if indent > indent_levels[-1]:
                if indent - indent_levels[-1] != 2:
                    raise SystemExit("Indent must increase by 2 spaces")
                if last_key_stack[-1] is None:
                    raise SystemExit("Indentation without a parent key")
                new_map = {}
                stack[-1][last_key_stack[-1]] = new_map
                stack.append(new_map); indent_levels.append(indent); last_key_stack.append(None)
            # key: value OR key: [a,b]
            if ':' not in line:
                raise SystemExit(f"Invalid line: {raw.rstrip()}")
            key, val = line.lstrip().split(':',1)
            key = key.strip()
            val = val.strip()
            if val == "":
                # will create nested map on next indent
                last_key_stack[-1] = key
                continue
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1].strip()
                items = [] if not inner else [coerce(x.strip()) for x in inner.split(',')]
                stack[-1][key] = items
                last_key_stack[-1] = None
            else:
                stack[-1][key] = coerce(val)
                last_key_stack[-1] = None
    return data

def sh(cmd):
    print('>>', ' '.join(map(str, cmd)))
    return subprocess.run(cmd, check=True).returncode

def resolve_wrapper(path_str: str) -> Path:
    cand = Path(path_str)
    if cand.exists():
        return cand
    # Try prefixed with 'validation/'
    cand2 = Path('validation') / path_str
    if cand2.exists():
        return cand2
    # If user passed 'run_validation.py', prefer 'validation/run_validation.py' if it exists
    if path_str == 'run_validation.py':
        cand3 = Path('validation') / 'run_validation.py'
        if cand3.exists():
            return cand3
    # Fallback to original
    return cand

def main(argv=None):
    ap = argparse.ArgumentParser(description='Run validation from YAML config (no external deps).')
    ap.add_argument('--config', required=True, help='Path to validation_config.yaml')
    ap.add_argument('--wrapper', default='validation/run_validation.py', help='Path to run_validation.py (default: validation/run_validation.py)')
    args = ap.parse_args(argv)

    cfg = parse_simple_yaml(args.config)
    b = cfg.get('builder', {})
    bt = cfg.get('backtester', {})
    p = cfg.get('paths', {})

    py    = p.get('python', sys.executable)
    wrap  = resolve_wrapper(args.wrapper)
    outdir = b.get('outdir', 'validation/inputs')

    cmd = [py, str(wrap)]
    # latest outputs logic
    if b.get('use_latest_outputs', False):
        cmd += ['--use-latest-outputs']
        if b.get('outputs_root'):
            cmd += ['--outputs-root', b['outputs_root']]
        if b.get('a_pattern'):
            cmd += ['--a-pattern', b['a_pattern']]
        if b.get('b_pattern'):
            cmd += ['--b-pattern', b['b_pattern']]
    # explicit A/B (if provided)
    if b.get('model_a_csv'):
        cmd += ['--model-a-csv', b['model_a_csv']]
    if b.get('model_b_csv'):
        cmd += ['--model-b-csv', b['model_b_csv']]

    # builder args
    cmd += ['--kraken-price-csv', b['kraken_price_csv']]
    cmd += ['--rebalance', b.get('rebalance','W-MON')]
    cmd += ['--N', str(b.get('N',10))]
    cmd += ['--cap', str(b.get('cap',0.25))]
    cmd += ['--floor-eur', str(b.get('floor_eur',25))]
    cmd += ['--portfolio-eur', str(b.get('portfolio_eur',1000))]
    cmd += ['--skip-days', str(b.get('skip_days',7))]
    cmd += ['--exp-uptrend', str(b.get('exp_uptrend',1.0))]
    cmd += ['--exp-range', str(b.get('exp_range',0.25))]
    cmd += ['--exp-downtrend', str(b.get('exp_downtrend',0.0))]
    if b.get('start'):
        cmd += ['--start', b['start']]
    if b.get('end'):
        cmd += ['--end', b['end']]
    cmd += ['--outdir', outdir]

    # backtester args
    cmd += ['--prices', bt['prices']]
    for bench in (bt.get('benches') or []):
        cmd += ['--bench', str(bench)]
    cmd += ['--signal-lag', str(bt.get('signal_lag',1))]
    cmd += ['--tc-bps', str(bt.get('tc_bps',10))]
    # Map run_tag to backtest outdir if no explicit outdir provided
    if bt.get('outdir'):
        cmd += ['--backtest-outdir', bt['outdir']]
    elif bt.get('run_tag'):
        cmd += ['--backtest-outdir', f"validation/runs/{bt['run_tag']}"]

    # script paths
    if p.get('builder_path'):
        cmd += ['--builder-path', p['builder_path']]
    if p.get('backtest_path'):
        cmd += ['--backtest-path', p['backtest_path']]

    return sh(cmd)

if __name__ == '__main__':
    raise SystemExit(main())
