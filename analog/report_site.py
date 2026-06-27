#!/usr/bin/env python3
"""
Phase 5 — Morning / Evening report generator for the website.

Surfaces the VALIDATED product (DESIGN.md §8.3): the forward VOLATILITY / RANGE
distribution implied by today's closest historical analogs. Direction is shown
only as a low-confidence context line, explicitly flagged as NOT validated (§8.1).

Modes:
  morning  — pre-market: "what to expect today/this week" from the last close
  evening  — post-close: EOD update + the same forward-range read for next sessions

Writes docs/data/<mode>_<name>.json, consumed by docs/index.html.

Usage:
  python analog/report_site.py --mode morning --name nifty
  python analog/report_site.py --mode auto           # all indices, mode by UTC hour
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config                       # noqa: E402
import engine as E                  # noqa: E402
import indicators as ind            # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
DOCS_DATA = os.path.join(config.REPO_ROOT, "docs", "data")
RANGE_TARGET = "fwd_hl_range_5"
VOL_TARGET = "fwd_rv_5"
LABELS = {"nifty": "NIFTY 50", "banknifty": "BANK NIFTY"}


def _pctile(baseline_vals, x):
    b = baseline_vals.dropna()
    return float((b < x).mean()) if len(b) else None


def _regime(pct):
    if pct is None:
        return "n/a"
    if pct >= 0.66:
        return "ELEVATED — expect a wider tape than usual"
    if pct <= 0.33:
        return "COMPRESSED — expect a quieter tape than usual"
    return "NORMAL"


def build(name, mode, df=None):
    df = df if df is not None else E.load(name)
    res = E.find_analogs(name, None, df=df)        # latest queryable bar
    q = res.query_idx
    close = float(df.loc[q, "close"])

    nb_range = res.neighbors[RANGE_TARGET].dropna()
    bl_range = res.baseline[RANGE_TARGET]
    nb_vol = res.neighbors[VOL_TARGET].dropna()
    bl_vol = res.baseline[VOL_TARGET]

    pred_range = float(nb_range.median())
    pred_range_p25, pred_range_p75 = float(nb_range.quantile(.25)), float(nb_range.quantile(.75))
    pred_range_q80 = float(nb_range.quantile(.80))
    base_range = float(bl_range.median())
    range_pct = _pctile(bl_range, pred_range)

    # next-session CPR from the latest bar's OHLC (DESIGN §3.6 convention)
    lb = df.loc[q]
    cpr = ind.cpr_levels(pd.Series([lb["high"]]), pd.Series([lb["low"]]),
                         pd.Series([lb["close"]])).iloc[0]

    # direction context (NOT validated — flagged)
    fwd5 = res.neighbors["fwd_ret_5"].dropna()
    dir_p_up = float((fwd5 > 0).mean()) if len(fwd5) else None

    report = {
        "index": name,
        "index_label": LABELS.get(name, name.upper()),
        "mode": mode,
        "generated_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "as_of_bar": str(df.loc[q, "date"].date()),
        "close": round(close, 2),
        "confidence": round(res.confidence, 2),
        "confidence_label": res.note,
        "pool_size": int(res.pool_size),
        "k": len(res.neighbors),

        "range_5d": {
            "predicted_pct": round(pred_range * 100, 2),
            "predicted_points": round(pred_range * close, 0),
            "p25_pct": round(pred_range_p25 * 100, 2),
            "p75_pct": round(pred_range_p75 * 100, 2),
            "q80_pct": round(pred_range_q80 * 100, 2),
            "baseline_pct": round(base_range * 100, 2),
            "vs_baseline_pp": round((pred_range - base_range) * 100, 2),
            "percentile": (None if range_pct is None else round(range_pct * 100, 0)),
            "regime": _regime(range_pct),
            "expected_low": round(close * (1 - pred_range / 2), 0),
            "expected_high": round(close * (1 + pred_range / 2), 0),
        },
        "realized_vol_5d": {
            "predicted_annual_pct": round(float(nb_vol.median()) * 100, 1),
            "baseline_annual_pct": round(float(bl_vol.median()) * 100, 1),
        },
        "cpr_next": {
            "pivot": round(float(cpr["pivot"]), 1),
            "tc": round(float(cpr["top"]), 1),
            "bc": round(float(cpr["bot"]), 1),
            "width_pct": round(abs(cpr["top"] - cpr["bot"]) / cpr["pivot"] * 100, 3),
        },
        "direction_context": {
            "p_up_5d": (None if dir_p_up is None else round(dir_p_up * 100, 0)),
            "disclaimer": "Direction is NOT validated out-of-sample (DESIGN.md §8.1) — "
                          "context only, do not trade on it.",
        },
        "precedents": [
            {"date": str(pd.Timestamp(r.date).date()),
             "realized_range_pct": (None if pd.isna(getattr(r, RANGE_TARGET))
                                    else round(getattr(r, RANGE_TARGET) * 100, 2)),
             "realized_ret_5d": (None if pd.isna(r.fwd_ret_5) else round(r.fwd_ret_5 * 100, 2))}
            for r in res.neighbors.head(8).itertuples()
        ],
    }
    return report


def write(report):
    os.makedirs(DOCS_DATA, exist_ok=True)
    out = os.path.join(DOCS_DATA, f"{report['mode']}_{report['index']}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return out


def resolve_mode(mode):
    if mode != "auto":
        return mode
    # morning if generated before ~12:00 UTC (covers the 02:45 UTC pre-market cron),
    # else evening (covers the 13:00 UTC post-close cron).
    return "morning" if datetime.now(timezone.utc).hour < 12 else "evening"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["morning", "evening", "auto"])
    ap.add_argument("--name", choices=list(config.SYMBOLS), help="default: all indices")
    a = ap.parse_args()

    mode = resolve_mode(a.mode)
    names = [a.name] if a.name else list(config.SYMBOLS)
    for name in names:
        rep = build(name, mode)
        out = write(rep)
        r = rep["range_5d"]
        print(f"[{mode}] {rep['index_label']} as of {rep['as_of_bar']} close {rep['close']:.0f} "
              f"-> 5d range {r['predicted_pct']}% ({r['regime'].split(' ')[0]}), "
              f"conf {rep['confidence']} [{rep['confidence_label']}] -> {out}")


if __name__ == "__main__":
    main()
