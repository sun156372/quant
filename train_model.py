#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py -- rebuild model/mvp_logit.pkl on the server.

It downloads the same Binance USDT-M 1H klines the research used
(data.binance.vision, no API key), builds the EXACT features from
strategy.add_features (single source of truth), fits the frozen M1 logistic
regression, and pickles it.

    python train_model.py                 # 4 core symbols, 3 years
    python train_model.py --symbols BTC   # single symbol
    python train_model.py --years 2

Why the server must run this rather than copy a .pkl:
  * the pickle stores the class path `hl_model.Logit`, so hl_model.py must exist
  * rebuilding from the same source guarantees the feature code and the model
    can never drift apart
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "_pylibs"):   # local dev: bundled scipy/lightgbm
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hl_model import Logit, save               # noqa: E402
from strategy import FEATURES, add_features    # noqa: E402

BASE = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1h/"
UA = {"User-Agent": "Mozilla/5.0"}
LABEL_THR = 0.005      # +0.5% ...
HORIZON = 4            # ... within 4 bars
LAM = 1.0


def months_back(years: int) -> list[str]:
    today = pd.Timestamp.now("UTC").normalize()
    start = today - pd.DateOffset(years=years)
    out = []
    for p in pd.date_range(start, today, freq="MS"):
        out.append(f"{p:%Y-%m}")
    return out


def download(sym: str, months: list[str], cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"{sym}_1h.parquet"
    if cache.exists():
        print(f"  {sym}: cache {cache}")
        return pd.read_parquet(cache)
    frames, missing = [], []
    for m in months:
        ok = False
        for attempt in range(3):
            try:
                url = BASE.format(s=sym) + f"{sym}-1h-{m}.zip"
                blob = urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=40).read()
                z = zipfile.ZipFile(io.BytesIO(blob))
                frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))))
                ok = True
                break
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        if not ok:
            missing.append(m)
    if not frames:
        raise RuntimeError(f"{sym}: no monthly files could be downloaded")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  {sym}: {len(df):,} bars" + (f"  (missing {missing})" if missing else ""))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC,ETH,SOL,BNB")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="model/mvp_logit.pkl")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    months = months_back(args.years)
    print(f"training on {syms}  |  {len(months)} months ({months[0]} -> {months[-1]})")

    rows = []
    for s in syms:
        raw = download(s, months, Path(args.data_dir))
        d = add_features(raw)                      # identical to live code
        d["fwd"] = d["close"].shift(-HORIZON) / d["close"] - 1
        d = d.dropna(subset=FEATURES + ["fwd"])
        rows.append(d[FEATURES + ["fwd"]])
    df = pd.concat(rows, ignore_index=True)
    y = (df["fwd"] >= LABEL_THR).astype(int).to_numpy()

    print(f"  pooled {len(df):,} rows  base rate {y.mean() * 100:.2f}%")
    model = Logit(LAM).fit(df[FEATURES].to_numpy(), y)
    p = model.predict(df[FEATURES].to_numpy())
    print(f"  in-sample mean P(up) {p.mean():.3f}  "
          f"min {p.min():.3f}  max {p.max():.3f}")
    print(f"  weights {np.round(model.w, 4).tolist()}")

    path = save(model, args.out)
    print(f"\n  saved -> {path.resolve()}  ({path.stat().st_size / 1024:.1f} KB)")
    print("  next: python run_live.py --once   (dry_run must stay true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
