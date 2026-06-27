#!/usr/bin/env /usr/bin/python3
"""
Phase 2 — Analog matching engine (DESIGN.md §4).

Given a query day, finds the most-similar PRIOR days via two layers:
  1. Snapshot k-NN  — weighted Euclidean over z-scored feature vectors
  2. Shape DTW      — Dynamic Time Warping over trailing W-bar paths
…then blends them. Strict point-in-time: a query at index q only ever matches
candidates at index < q, and DTW additionally embargoes windows that overlap the
query window (DESIGN.md §2).

Normalization is query-relative and PIT-safe: continuous features are z-scored
using mean/std computed ONLY over the candidate pool (days < q), then the same
stats are applied to the query row. No future data ever touches a query.

API:
    res = find_analogs("nifty", query_date="2026-06-25")
    res.neighbors  -> DataFrame (date, snap_dist, dtw_dist, blended, fwd_ret_*)
    res.query_state -> Series of the query day's features

CLI:
    /usr/bin/python3 nse_screener/analog/engine.py --name nifty
    /usr/bin/python3 nse_screener/analog/engine.py --name nifty --date 2020-03-23 --k 40
"""
import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config                      # noqa: E402
from features import (             # noqa: E402
    FEATURE_GROUPS, FEATURE_COLS, LABEL_COLS, CATEGORICAL,
)

# Core series used for the DTW shape match (trajectory similarity).
DTW_SERIES = ["close_ret", "rsi14", "adx14"]


# ---------------------------------------------------------------------------
# Design matrix: PIT-safe, weighted, with one-hot categoricals
# ---------------------------------------------------------------------------
def _per_feature_weights():
    """Group weight split equally across its member features, groups summing to 1."""
    gw = config.GROUP_WEIGHTS
    tot = sum(gw.values())
    w = {}
    for grp, cols in FEATURE_GROUPS.items():
        per = (gw[grp] / tot) / len(cols)
        for c in cols:
            w[c] = per
    return w


CONT_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL]
# observed categories (open_vs_cpr in {-1,0,1}; cpr_rel in 0..5)
CAT_LEVELS = {"open_vs_cpr": [-1.0, 0.0, 1.0], "cpr_rel": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}


def _build_weighted_matrix(cand, query):
    """
    cand: DataFrame of candidate rows (continuous + categorical feature cols).
    query: Series (one row) of the same cols.
    Returns (M_cand [n,d], v_query [d]) with PIT z-scoring + group weights baked in.
    Continuous: z = (x-mean_cand)/std_cand, scaled by sqrt(weight).
    Categorical: one-hot, scaled so a full class mismatch contributes `weight`
                 to squared distance (each one-hot col * sqrt(weight/2)).
    """
    w = _per_feature_weights()
    cols_cand, cols_q = [], []

    # --- continuous ---
    mu = cand[CONT_COLS].mean()
    sd = cand[CONT_COLS].std(ddof=0).replace(0, np.nan)
    zc = (cand[CONT_COLS] - mu) / sd
    zq = (query[CONT_COLS] - mu) / sd
    for c in CONT_COLS:
        scale = np.sqrt(w[c])
        cols_cand.append((zc[c].fillna(0).values) * scale)
        cols_q.append((0.0 if pd.isna(zq[c]) else zq[c]) * scale)

    # --- categorical (one-hot) ---
    for c, levels in CAT_LEVELS.items():
        scale = np.sqrt(w[c] / 2.0)
        for lv in levels:
            cols_cand.append((cand[c].values == lv).astype(float) * scale)
            cols_q.append((1.0 if query[c] == lv else 0.0) * scale)

    M = np.column_stack(cols_cand)
    v = np.array(cols_q, dtype=float)
    return M, v


# ---------------------------------------------------------------------------
# DTW (dependency-free, classic O(W^2) DP)
# ---------------------------------------------------------------------------
def _dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def _znorm_window(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass
class AnalogResult:
    name: str
    query_date: pd.Timestamp
    query_idx: int
    query_state: pd.Series
    neighbors: pd.DataFrame
    baseline: pd.DataFrame   # unconditional outcomes over the admissible pool (base rate)
    pool_size: int
    confidence: float
    note: str


def load(name):
    df = pd.read_parquet(config.features_path(name))
    df = df.sort_values("date").reset_index(drop=True)
    df["close_ret"] = df["close"].pct_change()
    return df


def _resolve_query_idx(df, query_date):
    if query_date is None:
        # last bar that has all features present (queryable "today")
        valid = df.dropna(subset=FEATURE_COLS)
        return int(valid.index.max())
    ts = pd.Timestamp(query_date)
    sub = df[df["date"] <= ts]
    if sub.empty:
        raise ValueError(f"no bar on/before {query_date}")
    return int(sub.index.max())


def find_analogs(name, query_date=None, k=None, window=None, alpha=None, df=None,
                 use_dtw=True):
    k = k or config.K_NEIGHBORS
    W = window or config.WINDOW
    alpha = config.BLEND_ALPHA if alpha is None else alpha
    if not use_dtw:
        alpha = 1.0   # snapshot-only (fast path for walk-forward validation)
    if df is None:
        df = load(name)

    q = _resolve_query_idx(df, query_date)
    qrow = df.loc[q]
    if qrow[FEATURE_COLS].isna().any():
        missing = [c for c in FEATURE_COLS if pd.isna(qrow[c])]
        raise ValueError(f"query {qrow['date'].date()} missing features {missing}")

    # ---- embargo (DESIGN.md §2.2): a candidate j is admissible only if
    #   (a) its DTW window [j-W+1, j] does NOT overlap the query window  -> j <= q-W
    #   (b) its outcome window [j+1, j+maxH] is fully realized before q   -> j <= q-maxH
    # The second guards against outcome look-ahead (a near-query candidate's
    # fwd_ret_5 would span days AFTER the query date). Embargo = max of both.
    embargo = max(W, max(config.HORIZONS), config.DRAWDOWN_HORIZON)
    j_max = q - embargo
    pool = df.iloc[:j_max + 1].dropna(subset=FEATURE_COLS).copy()
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} valid candidates before {qrow['date'].date()} "
                         f"(embargo {embargo})")

    # ---- snapshot distance ----
    M, v = _build_weighted_matrix(pool, qrow)
    snap = np.sqrt(((M - v) ** 2).sum(axis=1))
    pool = pool.assign(snap_dist=snap)

    # ---- DTW shape distance over the trailing W-bar paths ----
    if use_dtw:
        qpaths = {s: _znorm_window(df[s].iloc[q - W + 1:q + 1].values) for s in DTW_SERIES}
        dtw = np.full(len(pool), np.nan)
        for n_, j in enumerate(pool.index.values):
            if j - W + 1 < 0:
                continue
            d, ok = 0.0, True
            for s in DTW_SERIES:
                wv = df[s].iloc[j - W + 1:j + 1].values
                if np.isnan(wv).any():
                    ok = False
                    break
                d += _dtw(qpaths[s], _znorm_window(wv))
            dtw[n_] = d / len(DTW_SERIES) if ok else np.nan
        pool["dtw_dist"] = dtw
        # comparable DTW window for every candidate; drop rare early-warmup rows
        pool = pool[pool["dtw_dist"].notna()].copy()
        if len(pool) < k:
            raise ValueError(f"only {len(pool)} candidates with computable DTW window")
    else:
        pool["dtw_dist"] = np.nan

    # ---- blend (z-normalize each distance over the pool, combine) ----
    def _z(x):
        x = pd.Series(x)
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd and sd > 0 else x * 0.0

    if use_dtw and alpha < 1.0:
        pool["blended"] = (alpha * _z(pool["snap_dist"]).values
                           + (1 - alpha) * _z(pool["dtw_dist"]).values)
    else:
        pool["blended"] = _z(pool["snap_dist"]).values

    neigh = pool.nsmallest(k, "blended")
    out_cols = ["date", "snap_dist", "dtw_dist", "blended"] + LABEL_COLS
    neighbors = neigh[out_cols].reset_index(drop=True)
    # baseline = unconditional outcomes over the SAME admissible pool (the base rate
    # the analog set must beat). PIT-safe: pool is already all days <= q-embargo.
    baseline = pool[["date", "snap_dist"] + LABEL_COLS].reset_index(drop=True)

    conf, note = _confidence(pool, neigh)
    return AnalogResult(
        name=name, query_date=qrow["date"], query_idx=q,
        query_state=qrow[["date", "close"] + FEATURE_COLS],
        neighbors=neighbors, baseline=baseline, pool_size=len(pool),
        confidence=conf, note=note,
    )


def _confidence(pool, neigh):
    """
    Formal 0-1 confidence (DESIGN.md §12.5). Three components, each in [0,1]:

      tightness   = clip( (pool_med - neigh_med) / (pool_med - pool_min), 0, 1 )
                    0 when the nearest analogs are no closer than a typical pool day
                    (today is "unprecedented"); 1 when they are as close as the single
                    closest day in all of history.
      consistency = |2*P(up @ primary horizon) - 1|
                    0 at a coin-flip split, 1 when precedents unanimously agree.
      sample      = clip(n_realized_outcomes / k, 0, 1)   (usually 1; guards thin tails)

      confidence  = sample * (0.5*tightness + 0.5*consistency)

    A label is attached so a low score is never silently averaged away.
    """
    snap_all = pool["snap_dist"]
    pool_min, pool_med = snap_all.min(), snap_all.median()
    neigh_med = neigh["snap_dist"].median()
    tightness = float(np.clip((pool_med - neigh_med) / (pool_med - pool_min + 1e-12), 0, 1))

    fwd = neigh[f"fwd_ret_{config.HORIZONS[-1]}"].dropna()
    up = (fwd > 0).mean() if len(fwd) else 0.5
    consistency = abs(up - 0.5) * 2.0
    sample = float(np.clip(len(fwd) / len(neigh), 0, 1))

    conf = float(np.clip(sample * (0.5 * tightness + 0.5 * consistency), 0, 1))

    if tightness < 0.15:
        note = "WEAK — nearest analogs barely closer than average; today has no strong precedent"
    elif consistency < 0.2:
        note = "MIXED — precedents disagree on direction"
    elif conf >= 0.60:
        note = "HIGH"
    elif conf >= 0.35:
        note = "MODERATE"
    else:
        note = "LOW"
    return conf, note


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(config.SYMBOLS), default="nifty")
    ap.add_argument("--date", help="query date YYYY-MM-DD (default: latest queryable bar)")
    ap.add_argument("--k", type=int, default=config.K_NEIGHBORS)
    ap.add_argument("--window", type=int, default=config.WINDOW)
    ap.add_argument("--alpha", type=float, default=config.BLEND_ALPHA)
    a = ap.parse_args()

    res = find_analogs(a.name, a.date, a.k, a.window, a.alpha)
    print(f"\n{a.name.upper()} ANALOGS — query {res.query_date.date()} "
          f"(close {res.query_state['close']:.2f})")
    print(f"candidates before query, {a.k} nearest | confidence {res.confidence:.2f} [{res.note}]")
    print("\nquery state:")
    for c in FEATURE_COLS:
        print(f"  {c:<18} {res.query_state[c]:+.4f}")
    print("\noutcome distribution across neighbors:")
    for hh in config.HORIZONS:
        col = f"fwd_ret_{hh}"
        s = res.neighbors[col].dropna()
        print(f"  +{hh}d:  P(up)={ (s>0).mean()*100:4.0f}%   median={s.median()*100:+.2f}%   "
              f"mean={s.mean()*100:+.2f}%   [p25 {s.quantile(.25)*100:+.2f}, p75 {s.quantile(.75)*100:+.2f}]")
    print("\ntop 10 precedents:")
    show = res.neighbors.head(10).copy()
    show["date"] = show["date"].dt.date
    print(show[["date", "snap_dist", "dtw_dist", "blended", "fwd_ret_1", "fwd_ret_5"]]
          .to_string(index=False,
                     float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
