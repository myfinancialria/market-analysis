#!/usr/bin/env /usr/bin/python3
"""
Phase 4 — Walk-forward validation (DESIGN.md §8). The go/no-go credibility test.

For every query day in a held-out span we build the analog set using ONLY prior
data (the engine is point-in-time safe) and record the predicted P(up) at the
chosen horizon together with the REALIZED outcome for that day. Then we ask:

  1. Calibration  — when the engine says "65% up", does it come up ~65%?
  2. Brier skill  — does conditioning on analogs beat just predicting the
                    prevailing base rate (the null model)?
  3. Directional edge — accuracy and mean forward return of high-confidence
                    "up" calls vs the unconditional base rate.
  4. Regime stability — does any edge hold year by year, or only in bull runs?

Honesty clause (DESIGN.md §8): the verdict is printed verbatim, pass or fail.

Usage:
    /usr/bin/python3 nse_screener/analog/validate.py --name nifty
    /usr/bin/python3 nse_screener/analog/validate.py --name nifty --horizon 5 --step 3
    /usr/bin/python3 nse_screener/analog/validate.py --name nifty --blend    # use DTW (slower)
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config                       # noqa: E402
import engine as E                  # noqa: E402
from features import FEATURE_COLS   # noqa: E402


def walk_forward(name, start, horizon, step, use_dtw, k, thresh):
    df = E.load(name)
    col = f"fwd_ret_{horizon}"
    last_realizable = len(df) - 1 - horizon          # outcome must exist
    start_ts = pd.Timestamp(start)

    valid = df.dropna(subset=FEATURE_COLS)
    qidx = [i for i in valid.index
            if i <= last_realizable and df.loc[i, "date"] >= start_ts]
    qidx = qidx[::step]

    recs = []
    for n_, q in enumerate(qidx):
        try:
            res = E.find_analogs(name, df.loc[q, "date"], k=k, df=df, use_dtw=use_dtw)
        except ValueError:
            continue
        nb = res.neighbors[col].dropna()
        bl = res.baseline[col].dropna()
        if len(nb) == 0 or len(bl) == 0:
            continue
        realized = df.loc[q, col]
        recs.append({
            "date": df.loc[q, "date"],
            "pred_p_up": float((nb > 0).mean()),
            "pred_median": float(nb.median()),
            "base_p_up": float((bl > 0).mean()),
            "confidence": res.confidence,
            "label": res.note,
            "realized_ret": float(realized),
            "realized_up": int(realized > 0),
        })
        if (n_ + 1) % 100 == 0:
            print(f"  ... {n_+1}/{len(qidx)} queries", flush=True)
    return pd.DataFrame(recs)


def calibration(d, bins=(0, .45, .50, .55, .60, .65, .70, 1.01)):
    d = d.copy()
    d["bucket"] = pd.cut(d["pred_p_up"], bins=list(bins), right=False)
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        if len(g) == 0:
            continue
        rows.append((str(b), len(g), g["pred_p_up"].mean(), g["realized_up"].mean()))
    return rows


def verdict(d, horizon, thresh):
    n = len(d)
    base_rate = d["realized_up"].mean()
    print(f"\n=== WALK-FORWARD VALIDATION ({n} queries, horizon +{horizon}d) ===")
    print(f"span {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"| unconditional up-rate (base) = {base_rate*100:.1f}%")

    # 1) Brier skill vs the prevailing base rate (null model)
    brier_a = float(((d["pred_p_up"] - d["realized_up"]) ** 2).mean())
    brier_b = float(((d["base_p_up"] - d["realized_up"]) ** 2).mean())
    skill = 1 - brier_a / brier_b if brier_b > 0 else float("nan")
    print(f"\n1) Brier: analog={brier_a:.4f}  baseline={brier_b:.4f}  "
          f"skill={skill*100:+.1f}%  ({'analog better' if skill>0 else 'no improvement'})")

    # 2) Calibration
    print("\n2) Calibration (predicted P(up) bucket -> realized up-rate):")
    print(f"   {'bucket':<14}{'n':>6}{'pred':>8}{'realized':>10}")
    for b, cnt, pred, real in calibration(d):
        flag = "" if cnt < 15 else ("  ok" if abs(pred - real) <= 0.07 else "  off")
        print(f"   {b:<14}{cnt:>6}{pred*100:>7.0f}%{real*100:>9.0f}%{flag}")

    # 3) Directional edge: high-confidence "up" calls
    sig = d[(d["pred_p_up"] >= thresh)]
    hi = d[(d["pred_p_up"] >= thresh) & (d["confidence"] >= 0.5)]
    print(f"\n3) Directional edge at P(up)>={thresh:.0%}:")
    for label, s in [(f"P(up)>={thresh:.0%}", sig), ("  + confidence>=0.50", hi)]:
        if len(s) == 0:
            print(f"   {label:<22} no signals"); continue
        acc = s["realized_up"].mean()
        mret = s["realized_ret"].mean()
        # t-stat of signal mean return vs 0
        t = mret / (s["realized_ret"].std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else float("nan")
        print(f"   {label:<22} n={len(s):<4} up-rate={acc*100:4.1f}% "
              f"(base {base_rate*100:.1f}%)  mean fwd{horizon}={mret*100:+.2f}%  t={t:+.2f}")
    all_mean = d["realized_ret"].mean()
    print(f"   {'(all queries)':<22} n={len(d):<4} "
          f"mean fwd{horizon}={all_mean*100:+.2f}%")

    # 4) Regime stability by year (high-conf up calls)
    print("\n4) Regime stability — high-confidence up calls by year:")
    print(f"   {'year':<6}{'n_sig':>6}{'up-rate':>9}{'mean_ret':>10}{'all_ret':>10}")
    d2 = d.copy(); d2["year"] = d2["date"].dt.year
    for y, g in d2.groupby("year"):
        gs = g[(g["pred_p_up"] >= thresh) & (g["confidence"] >= 0.5)]
        if len(gs) == 0:
            print(f"   {y:<6}{0:>6}{'-':>9}{'-':>10}{g['realized_ret'].mean()*100:>9.2f}%")
            continue
        print(f"   {y:<6}{len(gs):>6}{gs['realized_up'].mean()*100:>8.0f}%"
              f"{gs['realized_ret'].mean()*100:>9.2f}%{g['realized_ret'].mean()*100:>9.2f}%")

    # measured verdict
    edge_dir = len(hi) >= 20 and hi["realized_up"].mean() > base_rate + 0.03
    edge_ret = len(hi) >= 20 and hi["realized_ret"].mean() > all_mean
    print("\n=== VERDICT ===")
    if skill > 0 and edge_dir and edge_ret:
        print("PASS (provisional): analog conditioning beats the base rate out-of-sample on")
        print("Brier skill, directional up-rate, and mean forward return for high-confidence")
        print("calls. Recommend Phase 5 (daily report) with confidence labels surfaced.")
    elif skill > 0 or edge_dir:
        print("MIXED: some signal (positive on part of Brier/direction/return) but not all")
        print("three. Treat outputs as context, NOT a standalone signal. Candidate fixes:")
        print("re-weight feature groups, tune k, or restrict to high-confidence regimes.")
    else:
        print("FAIL: no reliable out-of-sample edge over the base rate. Per DESIGN.md §8 the")
        print("honest move is to revise the feature set (documented) or shelve the signal —")
        print("do NOT ship a 'prediction' the data does not support.")
    return dict(n=n, base_rate=base_rate, brier_skill=skill,
                hi_n=len(hi), hi_up=(hi['realized_up'].mean() if len(hi) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), default="nifty")
    ap.add_argument("--start", default="2018-01-01", help="test-span start date")
    ap.add_argument("--horizon", type=int, default=5, choices=config.HORIZONS)
    ap.add_argument("--step", type=int, default=3, help="trading days between queries")
    ap.add_argument("--k", type=int, default=config.K_NEIGHBORS)
    ap.add_argument("--thresh", type=float, default=0.60, help="P(up) signal threshold")
    ap.add_argument("--blend", action="store_true", help="use DTW blend (slower); default snapshot-only")
    a = ap.parse_args()

    print(f"Walk-forward {a.name} from {a.start}, horizon +{a.horizon}d, step {a.step}, "
          f"{'BLEND(+DTW)' if a.blend else 'snapshot-only (fast)'} ...", flush=True)
    d = walk_forward(a.name, a.start, a.horizon, a.step, a.blend, a.k, a.thresh)
    if len(d) == 0:
        print("no queries produced — check span/data"); return
    verdict(d, a.horizon, a.thresh)


if __name__ == "__main__":
    main()
