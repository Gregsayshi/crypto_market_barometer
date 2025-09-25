#!/usr/bin/env python3
# list_kraken_eur_pairs.py — produce pairs_kraken_eur.csv listing all EUR-quoted pairs on Kraken
# Usage:
#   python data/list_kraken_eur_pairs.py --out data/kraken/pairs_kraken_eur.csv

import argparse, csv, requests
from pathlib import Path

API = "https://api.kraken.com/0/public/AssetPairs"

def fetch_assetpairs() -> dict:
    r = requests.get(API, timeout=20)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(f"Kraken API error: {js['error']}")
    return js.get("result", {})

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List all Kraken EUR-quoted pairs into a CSV.")
    ap.add_argument("--out", default="data/kraken/pairs_kraken_eur.csv", help="Output CSV path")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    res = fetch_assetpairs()

    rows = []
    for key, meta in res.items():
        if not isinstance(meta, dict):
            continue
        altname = meta.get("altname", "")
        wsname  = meta.get("wsname", "")
        base    = meta.get("base", "")
        quote   = meta.get("quote", "")
        if quote == "ZEUR":  # EUR quote
            rows.append({"altname": altname, "wsname": wsname, "base": base, "quote": quote, "pair": key})

    rows.sort(key=lambda r: r["altname"])
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["altname","wsname","base","quote","pair"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} EUR-quoted pairs to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())