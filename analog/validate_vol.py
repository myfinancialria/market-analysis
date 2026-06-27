#!/usr/bin/env /usr/bin/python3
"""
Phase 4b — Walk-forward validation of VOLATILITY / RANGE prediction (DESIGN.md §8.2).

Direction failed OOS (§8.1). The honest strength of an analog engine is forecasting
the *distribution* — how wide the next few days will be — because volatility clusters.
This validator tests that, against the baseline that actually matters:

  * climatology  — predict the unconditional median forward range (the dumb prior)
  * PERSISTENCE  — predict forward range = trailing range (vol is autocorrelated; this
                   is the hard baseline the analog MUST beat to add value)
  * analog       — median forward range of the k nearest historical analogs

Metrics: Spearman rank-corr vs realized, MAE (+ skill = 1 - MAE_analog/MAE_persist),
tercile monotonicity, and quantile calibration (does the predicted 80th-pct band
contain ~80% of outcomes?). Verdict printed verbatim, pass or fail.

Usage:
    /usr/bin/python3 nse_screener/analog/validate_vol.py --name nifty
    /usr/bin/python3 nse_screener/analog/validate_vol.py --name nifty --target fwd_rv_5 --blend
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


def _spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank()))


def walk_forward(name, start, target, step, use_dtw, k):
    df = E.load(name)
    Hh = config.DRAWDOWN_HORIZON
    last_realizable = len(df) - 1 - Hh
    start_ts = pd.Timestamp(start)

    valid = df.dropna(subset=FEATURE_COLS)
    qidx = [i for i in valid.index
            if i <= last_realizable and df.loc[i, "date"] >= start_ts][::step]

    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    recs = []
    for n_, q in enumerate(qidx):
        try:
            res = E.find_analogs(name, df.loc[q, "date"], k=k, df=df, use_dtw=use_dtw)
        except ValueError:
            continue
        nb = res.neighbors[target].dropna()
        bl = res.baseline[target].dropna()
        if len(nb) < 5 or len(bl) < 20:
            continue
        realized = df.loc[q, target]
        if pd.isna(realized):
            continue
        # persistence predictor: trailing Hh-day realized range as % of close (PIT)
        trail = (hi[q - Hh + 1:q + 1].max() - lo[q - Hh + 1:q + 1].min()) / cl[q]
        recs.append({
            "date": df.loc[q, "date"],
            "analog_pred": float(nb.median()),
            "analog_q80": float(nb.quantile(0.80)),
            "climatology": float(bl.median()),
            "persistence": float(trail),
            "realized": float(realized),
        })
        if (n_ + 1) % 100 == 0:
            print(f"  ... {n_+1}/{len(qidx)} queries", flush=True)
    return pd.DataFrame(recs)


def verdict(d, target):
    n = len(d)
    print(f"\n=== VOL/RANGE WALK-FORWARD ({n} queries, target {target}) ===")
    print(f"span {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"| realized mean={d['realized'].mean():.4f} median={d['realized'].median():.4f}")

    # 1) rank correlation vs realized
    print("\n1) Spearman rank-corr with realized (higher = better ordering):")
    for col in ["analog_pred", "persistence", "climatology"]:
        print(f"   {col:<14} rho = {_spearman(d[col], d['realized']):+.3f}")

    # 2) MAE + skill vs persistence
    print("\n2) Mean absolute error (lower = better):")
    mae = {c: float((d[c] - d["realized"]).abs().mean())
           for c in ["analog_pred", "persistence", "climatology"]}
    for c, v in mae.items():
        print(f"   {c:<14} MAE = {v:.4f}")
    skill_p = 1 - mae["analog_pred"] / mae["persistence"]
    skill_c = 1 - mae["analog_pred"] / mae["climatology"]
    print(f"   -> skill vs persistence = {skill_p*100:+.1f}%   "
          f"vs climatology = {skill_c*100:+.1f}%")

    # 3) tercile monotonicity (does higher predicted range -> higher realized?)
    print("\n3) Analog-prediction terciles -> realized range (want monotonic increase):")
    d2 = d.copy()
    d2["tercile"] = pd.qcut(d2["analog_pred"], 3, labels=["low", "mid", "high"])
    means = d2.groupby("tercile", observed=True)["realized"].agg(["mean", "count"])
    for t, row in means.iterrows():
        print(f"   {t:<5} n={int(row['count']):<4} realized mean = {row['mean']:.4f}")
    monotonic = means["mean"].is_monotonic_increasing

    # 4) quantile calibration: realized should exceed the predicted q80 ~20% of time
    exceed = float((d["realized"] > d["analog_q80"]).mean())
    print(f"\n4) Predicted 80th-pct band: realized exceeded it {exceed*100:.1f}% of time "
          f"(target ~20%)")

    # verdict
    print("\n=== VERDICT ===")
    beats_persist = skill_p > 0.02 and _spearman(d["analog_pred"], d["realized"]) > \
        _spearman(d["persistence"], d["realized"])
    if beats_persist and monotonic:
        print("PASS: analog range-forecast beats the persistence baseline on MAE AND rank")
        print("ordering, and predicted terciles are monotonic. This is a shippable, honest")
        print("desk tool — 'days like today historically ran a ~X% 5-day range'. Proceed to")
        print("Phase 5 (daily report) surfacing range/vol, not direction.")
    elif skill_p > 0 or monotonic:
        print("MARGINAL: analog tracks volatility (vol clusters) but barely beats simple")
        print("persistence. Usable as confirmation/context, but persistence alone is nearly")
        print("as good. Consider richer vol features (GARCH-style, vol-of-vol) before shipping.")
    else:
        print("FAIL: analog adds nothing over persistence — it's just re-deriving 'today's vol")
        print("predicts tomorrow's vol'. Per §8, do not ship it as a distinct signal.")
    print(f"\n(note: persistence is a strong baseline — beating it even slightly is the real bar,")
    print(f" not beating climatology, which any vol-aware model trivially clears.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), default="nifty")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--target", default="fwd_hl_range_5", choices=["fwd_hl_range_5", "fwd_rv_5"])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--k", type=int, default=config.K_NEIGHBORS)
    ap.add_argument("--blend", action="store_true", help="use DTW blend (slower)")
    a = ap.parse_args()

    print(f"Vol walk-forward {a.name} from {a.start}, target {a.target}, step {a.step}, "
          f"{'BLEND(+DTW)' if a.blend else 'snapshot-only'} ...", flush=True)
    d = walk_forward(a.name, a.start, a.target, a.step, a.blend, a.k)
    if len(d) == 0:
        print("no queries produced"); return
    verdict(d, a.target)


if __name__ == "__main__":
    main()
