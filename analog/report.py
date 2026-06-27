#!/usr/bin/env /usr/bin/python3
"""
Phase 3 — Report layer for the Market Analog Engine (DESIGN.md §5).

Turns an AnalogResult into:
  - a per-horizon outcome distribution (analog set)
  - the unconditional BASELINE distribution over the same admissible pool (base rate)
  - the LIFT of the analog set over the baseline, with a proportion z-test so we can
    tell signal from noise
  - worst-decile drawdown (tail risk the analogs lived through)
  - the formal confidence score + label
…and writes it to data/analog_report_<name>_<date>.json plus a console summary.

Usage:
    /usr/bin/python3 nse_screener/analog/report.py --name nifty
    /usr/bin/python3 nse_screener/analog/report.py --name nifty --date 2020-03-23 --json-only
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config                       # noqa: E402
import engine as E                  # noqa: E402
from features import FEATURE_COLS   # noqa: E402


def _dist(series):
    s = series.dropna()
    if len(s) == 0:
        return dict(n=0, p_up=None, mean=None, median=None, p25=None, p75=None)
    return dict(
        n=int(len(s)),
        p_up=float((s > 0).mean()),
        mean=float(s.mean()),
        median=float(s.median()),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75)),
    )


def _prop_z(p_analog, n_analog, p_base):
    """Two-proportion-style z: is the analog up-rate different from the base rate?
    Uses the base rate as the null proportion (its n is large vs the analog k)."""
    if p_analog is None or p_base is None or n_analog == 0:
        return None
    se = np.sqrt(p_base * (1 - p_base) / n_analog)
    if se == 0:
        return None
    return float((p_analog - p_base) / se)


def build_report(res):
    horizons = config.HORIZONS
    rows = {}
    for hh in horizons:
        col = f"fwd_ret_{hh}"
        a = _dist(res.neighbors[col])
        b = _dist(res.baseline[col])
        z = _prop_z(a["p_up"], a["n"], b["p_up"])
        rows[f"+{hh}d"] = {
            "analog": a,
            "baseline": b,
            "lift_p_up_pp": (None if a["p_up"] is None or b["p_up"] is None
                             else round((a["p_up"] - b["p_up"]) * 100, 1)),
            "lift_median_pp": (None if a["median"] is None or b["median"] is None
                               else round((a["median"] - b["median"]) * 100, 2)),
            "prop_z": (None if z is None else round(z, 2)),
            "significant_90pct": (None if z is None else bool(abs(z) >= 1.645)),
        }

    tail = res.neighbors["fwd_maxdd_5"].dropna()
    tail_up = res.neighbors["fwd_maxup_5"].dropna()

    report = {
        "engine": "market-analog",
        "index": res.name,
        "query_date": str(res.query_date.date()),
        "query_close": float(res.query_state["close"]),
        "pool_size": int(res.pool_size),
        "k_neighbors": len(res.neighbors),
        "confidence": round(res.confidence, 3),
        "confidence_label": res.note,
        "query_state": {c: float(res.query_state[c]) for c in FEATURE_COLS},
        "horizons": rows,
        "tail_risk_5d": {
            "worst_decile_drawdown": (None if len(tail) == 0 else float(tail.quantile(0.10))),
            "median_max_drawup": (None if len(tail_up) == 0 else float(tail_up.median())),
        },
        "precedents": [
            {"date": str(pd.Timestamp(r.date).date()),
             "snap_dist": round(float(r.snap_dist), 4),
             "dtw_dist": (None if pd.isna(r.dtw_dist) else round(float(r.dtw_dist), 4)),
             "blended": round(float(r.blended), 4),
             "fwd_ret_1": (None if pd.isna(r.fwd_ret_1) else round(float(r.fwd_ret_1), 4)),
             "fwd_ret_5": (None if pd.isna(r.fwd_ret_5) else round(float(r.fwd_ret_5), 4))}
            for r in res.neighbors.head(10).itertuples()
        ],
    }
    return report


def print_report(rep):
    print(f"\n{rep['index'].upper()} ANALOG REPORT — {rep['query_date']} "
          f"(close {rep['query_close']:.2f})")
    print(f"pool={rep['pool_size']} candidates | k={rep['k_neighbors']} | "
          f"confidence {rep['confidence']:.2f} [{rep['confidence_label']}]")

    print("\noutcome distribution — ANALOG vs BASELINE (base rate):")
    print(f"  {'horizon':<7} {'P(up)':>14} {'lift':>7} {'median':>15} {'z':>6}  signif")
    for h, d in rep["horizons"].items():
        a, b = d["analog"], d["baseline"]
        if a["p_up"] is None:
            continue
        sig = "" if d["significant_90pct"] is None else ("**" if d["significant_90pct"] else "")
        print(f"  {h:<7} {a['p_up']*100:5.0f}% vs {b['p_up']*100:4.0f}%"
              f" {d['lift_p_up_pp']:+5.1f}pp"
              f"  {a['median']*100:+6.2f}% vs {b['median']*100:+5.2f}%"
              f" {d['prop_z']:+5.2f}  {sig}")

    t = rep["tail_risk_5d"]
    if t["worst_decile_drawdown"] is not None:
        print(f"\n5-day tail: worst-decile drawdown {t['worst_decile_drawdown']*100:+.2f}%, "
              f"median max draw-up {t['median_max_drawup']*100:+.2f}%")

    print("\ntop precedents:")
    for p in rep["precedents"]:
        fr5 = "   n/a" if p["fwd_ret_5"] is None else f"{p['fwd_ret_5']*100:+5.2f}%"
        print(f"  {p['date']}  blended {p['blended']:+.3f}  fwd5 {fr5}")
    print("\n** = analog up-rate differs from base rate at 90% (proportion z-test); "
          "NOT out-of-sample proof — see validate.py / DESIGN.md §8.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), default="nifty")
    ap.add_argument("--date", help="query date YYYY-MM-DD (default: latest queryable bar)")
    ap.add_argument("--k", type=int, default=config.K_NEIGHBORS)
    ap.add_argument("--json-only", action="store_true")
    a = ap.parse_args()

    res = E.find_analogs(a.name, a.date, a.k)
    rep = build_report(res)

    out = os.path.join(config.DATA_DIR, f"analog_report_{a.name}_{rep['query_date']}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)

    if not a.json_only:
        print_report(rep)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
