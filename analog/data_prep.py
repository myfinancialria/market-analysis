#!/usr/bin/env /usr/bin/python3
"""
Phase 0 — Data prep for the Market Analog Engine.

Pulls full-history daily OHLCV for each index from yfinance (the single
authoritative source) and writes a clean, consistent parquet per index:

    data/analog_<name>_daily.parquet   columns: date, open, high, low, close, volume

yfinance gives longer + more current history than the Fyers bt_*.parquet files
(^NSEI from 2007-09, refreshed daily). Index volume is unreliable/zero before
~2013 (Nifty) / ~2011 (Bank Nifty); we keep it as-is and let the feature layer
flag/NaN volume features where it is missing.

Usage:
    /usr/bin/python3 nse_screener/analog/data_prep.py            # all symbols
    /usr/bin/python3 nse_screener/analog/data_prep.py --name nifty
"""
import os
import sys
import argparse

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402


def fetch(symbol):
    """Full-history daily OHLCV from yfinance, tz-naive, sorted, de-duped."""
    h = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    h.index = h.index.tz_localize(None)
    h = h.rename(columns=str.lower)
    h.index.name = "date"
    h = h.reset_index()
    h = h.dropna(subset=["open", "high", "low", "close"])
    h = h.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    h["volume"] = h["volume"].fillna(0).astype("int64")
    return h


def save(name, df):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out = config.price_path(name)
    df.to_parquet(out, index=False)
    return out


def report(name, df):
    v = df["volume"]
    vnz = v[v > 0]
    print(f"\n[{name}]  {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  volume nonzero: {int((v > 0).sum())}/{len(v)} rows"
          + (f"  (from {vnz.index.min() and df.loc[vnz.index.min(), 'date'].date()})" if len(vnz) else "  (none)"))
    # gap sanity: largest run of calendar days between consecutive bars
    gaps = df["date"].diff().dt.days.dropna()
    big = gaps[gaps > 7]
    print(f"  max gap between bars: {int(gaps.max())} calendar days; gaps>7d: {len(big)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), help="single index (default: all)")
    args = ap.parse_args()

    names = [args.name] if args.name else list(config.SYMBOLS)
    for name in names:
        sym = config.SYMBOLS[name]
        print(f"Fetching {name} ({sym}) ...", flush=True)
        df = fetch(sym)
        out = save(name, df)
        report(name, df)
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
