#!/usr/bin/env /usr/bin/python3
"""
Phase 1 — Feature library for the Market Analog Engine (DESIGN.md §3).

Reads data/analog_<name>_daily.parquet (from data_prep.py) and writes a
per-bar feature table data/features_<name>.parquet.

Every feature at bar t is point-in-time safe (uses only bars <= t). The CPR
features use day t-1's OHLC for the CPR "active" on day t, exactly as the rest
of the repo does. Forward returns are LABELS (never features) and use t+1..t+5.

Usage:
    /usr/bin/python3 nse_screener/analog/features.py            # all symbols
    /usr/bin/python3 nse_screener/analog/features.py --name nifty
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config       # noqa: E402
import indicators as ind  # noqa: E402


# Feature -> group, for the engine's weighted distance (DESIGN.md §4.1).
FEATURE_GROUPS = {
    "trend":      ["px_vs_50dma", "px_vs_200dma", "dma_spread", "dma50_slope"],
    "momentum":   ["rsi14", "rsi14_slope"],
    "strength":   ["adx14", "di_gap"],
    "volume":     ["vol_ratio", "updown_vol"],
    "volatility": ["atr_pct", "dist_20d_high", "dist_20d_low"],
    "cpr":        ["cpr_width_pct", "cpr_width_pctile", "open_vs_cpr", "cpr_rel"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
LABEL_COLS = ([f"fwd_ret_{h}" for h in config.HORIZONS]
              + ["fwd_maxdd_5", "fwd_maxup_5", "fwd_hl_range_5", "fwd_rv_5"])
CATEGORICAL = {"open_vs_cpr", "cpr_rel"}  # not z-scored by the engine


def _rolling_pctile_rank(s, n):
    """Rank of the current value within its trailing n-bar window, in [0,1]."""
    return s.rolling(n).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def build(df):
    """df: date, open, high, low, close, volume (sorted). Returns feature frame."""
    df = df.sort_values("date").reset_index(drop=True)
    o, h, l, c, v = (df[x] for x in ("open", "high", "low", "close", "volume"))

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()

    # --- trend / location -------------------------------------------------
    ma50 = ind.sma(c, config.DMA_FAST)
    ma200 = ind.sma(c, config.DMA_SLOW)
    out["px_vs_50dma"] = c / ma50 - 1
    out["px_vs_200dma"] = c / ma200 - 1
    out["dma_spread"] = ma50 / ma200 - 1
    out["dma50_slope"] = ma50.pct_change(config.SLOPE_LOOKBACK)

    # --- momentum ---------------------------------------------------------
    r = ind.rsi(c, config.RSI_N)
    out["rsi14"] = r
    out["rsi14_slope"] = r - r.shift(config.SLOPE_LOOKBACK)

    # --- trend strength (ADX) --------------------------------------------
    adx_df = ind.adx(h, l, c, config.ADX_N)
    out["adx14"] = adx_df["adx"]
    out["di_gap"] = adx_df["plus_di"] - adx_df["minus_di"]

    # --- volume -----------------------------------------------------------
    vol_avg = v.rolling(config.VOL_AVG_N).mean()
    out["vol_ratio"] = v / vol_avg.replace(0, np.nan)
    signed_vol = v * np.sign(c.diff())
    out["updown_vol"] = (signed_vol.rolling(10).sum()
                         / v.rolling(10).sum().replace(0, np.nan))
    # index volume is missing/zero pre-2013 (Nifty) / 2011 (BankNifty): NaN it.
    no_vol = vol_avg.isna() | (vol_avg == 0)
    out.loc[no_vol, ["vol_ratio", "updown_vol"]] = np.nan

    # --- volatility / structure ------------------------------------------
    out["atr_pct"] = ind.atr(h, l, c, config.ATR_N) / c
    out["dist_20d_high"] = c / h.rolling(config.HIGHLOW_N).max() - 1
    out["dist_20d_low"] = c / l.rolling(config.HIGHLOW_N).min() - 1

    # --- CPR (active on day t = built from day t-1 OHLC) -----------------
    cpr_raw = ind.cpr_levels(h, l, c)          # row t built from bar t
    cpr_act = cpr_raw.shift(1)                  # CPR active on bar t
    width = (cpr_act["top"] - cpr_act["bot"]).abs() / cpr_act["pivot"]
    out["cpr_width_pct"] = width
    out["cpr_width_pctile"] = _rolling_pctile_rank(width, config.CPR_WIDTH_PCTILE_N)
    out["open_vs_cpr"] = np.where(o > cpr_act["top"], 1.0,
                          np.where(o < cpr_act["bot"], -1.0, 0.0))
    out.loc[cpr_act["top"].isna(), "open_vs_cpr"] = np.nan
    out["cpr_rel"] = ind.cpr_relationship_series(cpr_act)

    # --- forward labels (NOT features) -----------------------------------
    for hh in config.HORIZONS:
        out[f"fwd_ret_{hh}"] = c.shift(-hh) / c - 1
    # --- window labels over the next DRAWDOWN_HORIZON days ----------------
    # full_window: the furthest bar (t+Hh) must exist, else the window is only
    # partially in the future -> label is NaN (avoids pandas skipna using a
    # truncated window at the dataset tail).
    Hh = config.DRAWDOWN_HORIZON
    full_window = c.shift(-Hh).notna()

    fwd = pd.concat([c.shift(-i) / c - 1 for i in range(1, Hh + 1)], axis=1)
    out["fwd_maxdd_5"] = fwd.min(axis=1).where(full_window)
    out["fwd_maxup_5"] = fwd.max(axis=1).where(full_window)
    # forward realized RANGE (intraday high-low span over next Hh days, % of close)
    fmax = pd.concat([h.shift(-i) for i in range(1, Hh + 1)], axis=1).max(axis=1)
    fmin = pd.concat([l.shift(-i) for i in range(1, Hh + 1)], axis=1).min(axis=1)
    out["fwd_hl_range_5"] = ((fmax - fmin) / c).where(full_window)
    # forward realized VOLATILITY (annualized std of close-to-close daily returns)
    rets = pd.concat([c.shift(-i) / c.shift(-(i - 1)) - 1 for i in range(1, Hh + 1)], axis=1)
    out["fwd_rv_5"] = (rets.std(axis=1, ddof=1) * np.sqrt(252)).where(full_window)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), help="single index (default: all)")
    args = ap.parse_args()

    names = [args.name] if args.name else list(config.SYMBOLS)
    for name in names:
        src = config.price_path(name)
        if not os.path.exists(src):
            print(f"[{name}] missing {src} — run data_prep.py first"); continue
        df = pd.read_parquet(src)
        feat = build(df)
        out = config.features_path(name)
        feat.to_parquet(out, index=False)

        # coverage report
        usable = feat.dropna(subset=FEATURE_COLS)
        print(f"\n[{name}] {len(feat)} bars  {feat['date'].min().date()} -> {feat['date'].max().date()}")
        print(f"  rows with ALL features present: {len(usable)}"
              f"  (first fully-featured bar: {usable['date'].min().date() if len(usable) else 'n/a'})")
        nan_pct = feat[FEATURE_COLS].isna().mean().mul(100).round(1)
        print("  %NaN by feature (warmup/volume gaps):")
        for col, pct in nan_pct.items():
            print(f"    {col:<18} {pct:5.1f}%")
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
