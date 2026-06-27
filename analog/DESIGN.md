# Market Analog Engine — Design Document

**Author:** Research/Quant desk
**Status:** Design (pre-implementation)
**Scope:** Nifty 50 & Bank Nifty, daily bars, 1–5 day forward horizon
**Last updated:** 2026-06-27

---

## 1. Purpose & framing

Build a **precedent engine**, not a predictor.

Given the market's state *today*, the engine answers one question:

> "Today's market looks most like these N past sessions. Here is the distribution
> of what actually happened over the next 1, 3, and 5 days — odds of an up move,
> median return, the drawdown you'd have stomached, and the actual historical dates
> so you can pull up the charts yourself."

This framing is deliberate and load-bearing:

- **Explainable.** Every output traces back to real, dated historical sessions. We can
  show a client or compliance officer exactly *why* the engine said what it said.
- **Honest about uncertainty.** We report a *distribution* of outcomes, never a single
  point forecast. "70% up, median +0.8%, worst-decile −2.1%" is a defensible statement.
  "Nifty will go up tomorrow" is not.
- **Hard to overfit.** A k-NN / analog approach has no trainable weights to memorize the
  past. It either finds genuine precedents or it doesn't, and we measure that honestly
  (Section 8).

What this engine is **not**: a black-box buy/sell signal, a guarantee, or a substitute
for risk management. It is a probabilistic context tool for a trading/advisory desk.

---

## 2. Two non-negotiable correctness rules

These are the difference between a real research tool and a backtest that lies to you.

### 2.1 Point-in-time (no look-ahead)
A query day `t` may only be matched against historical days **strictly earlier than `t`**.
Every feature for day `t` must be computable using information available *at or before the
close of `t`* (or, for the open-based CPR features, at the open of `t`). No feature may peek
at day `t+1`.

- CPR for day `t` is built from day `t-1`'s OHLC → valid. (Already true in existing CPR code.)
- 50/200 DMA, RSI, ADX, ATR use trailing windows ending at `t` → valid.
- Forward returns (the *outcome* we measure) use `t+1 … t+5` → these are **labels**, never features.

### 2.2 Window embargo (no leakage between query and neighbors)
A candidate day `j` is admissible for a query at day `t` only if **both** hold:
- **Shape non-overlap**: its DTW window `[j-W+1, j]` does not overlap the query window
  `[t-W+1, t]` → `j ≤ t-W`. Without this, a 10-day window starting at `t-3` would "match" itself.
- **Outcome realized before the query**: its forward-return labels `[j+1, j+maxH]` must close
  on or before `t` → `j ≤ t-maxH`. A candidate at `t-2` has a `fwd_ret_5` spanning `t+3`, i.e.
  days *after* the query — using its outcome to forecast would be look-ahead.

**Embargo = `max(W, maxH)` trading days** before the query (implemented in `engine.py`). This
single rule covers both shape-overlap and outcome look-ahead, and incidentally removes the
serial-correlation artifacts of near-query days dominating the neighbor set.

---

## 3. Feature specification

All features are computed per daily bar and stored in `data/features_<index>.parquet`.
Formulas below are exact so the implementation is unambiguous. All reuse existing repo
code where it exists (see Section 6); **ADX is the only genuinely new indicator**.

### 3.1 Trend / location
| Feature | Formula | Notes |
|---|---|---|
| `px_vs_50dma` | `close / SMA(close, 50) - 1` | % distance above/below 50DMA |
| `px_vs_200dma` | `close / SMA(close, 200) - 1` | % distance above/below 200DMA |
| `dma_spread` | `SMA(close,50) / SMA(close,200) - 1` | >0 golden-cross regime, <0 death-cross |
| `dma50_slope` | `SMA(close,50).pct_change(5)` | 5-day slope of the 50DMA |

### 3.2 Momentum
| Feature | Formula | Notes |
|---|---|---|
| `rsi14` | Wilder RSI, n=14 (ewm of gains/losses) | reuse `patterns.py` impl |
| `rsi14_slope` | `rsi14 - rsi14.shift(5)` | rising/falling momentum |

### 3.3 Trend strength — **NEW (ADX)**
Wilder's ADX(14). Build once, reuse everywhere.
```
TR    = max(H-L, |H-Cprev|, |L-Cprev|)
+DM   = (H-Hprev) if (H-Hprev) > (Lprev-L) and (H-Hprev) > 0 else 0
-DM   = (Lprev-L) if (Lprev-L) > (H-Hprev) and (Lprev-L) > 0 else 0
ATR14 = Wilder-smooth(TR, 14)
+DI14 = 100 * Wilder-smooth(+DM,14) / ATR14
-DI14 = 100 * Wilder-smooth(-DM,14) / ATR14
DX    = 100 * |+DI14 - -DI14| / (+DI14 + -DI14)
ADX14 = Wilder-smooth(DX, 14)
```
| Feature | Source |
|---|---|
| `adx14` | ADX14 above (trend strength, 0–100) |
| `di_gap` | `+DI14 - -DI14` (directional bias, signed) |

Wilder smoothing = `ewm(alpha=1/14, adjust=False)` — *not* a simple rolling mean.

### 3.4 Volume
| Feature | Formula | Notes |
|---|---|---|
| `vol_ratio` | `volume / SMA(volume, 50)` | participation surge/dry-up |
| `updown_vol` | rolling sum of (volume on up days − down days) / total, 10d | crude up/down volume balance |

> Index volume: `^NSEI`/`^NSEBANK` from yfinance carry volume; if unreliable we fall back
> to a constituent-aggregate or drop volume features for the index variant (flagged at build).

### 3.5 Volatility / structure
| Feature | Formula | Notes |
|---|---|---|
| `atr_pct` | `ATR(20) / close` | normalized volatility, reuse `swing_backtest.py` ATR |
| `dist_20d_high` | `close / rolling(20).max() - 1` | proximity to recent high (≤0) |
| `dist_20d_low` | `close / rolling(20).min() - 1` | proximity to recent low (≥0) |

### 3.6 CPR (reuse existing taxonomy)
| Feature | Formula | Notes |
|---|---|---|
| `cpr_width_pct` | `(TC - BC).abs() / P` (from day t-1 OHLC) | narrow CPR = breakout potential |
| `cpr_width_pctile` | rolling 252-day percentile rank of `cpr_width_pct` | regime-relative width |
| `open_vs_cpr` | open above TC / inside / below BC → {+1, 0, −1} | needs day-t open (known at open) |
| `cpr_rel` | prior-day CPR relationship: Higher/Lower/Inside/Outside/Overlap | reuse `nifty_cpr_relationship.py` |

`open_vs_cpr` is the one feature that uses the day-`t` open. It is available at the open of
`t`, so a forecast made *at the open* may use it; a forecast made on the *prior close* must not.
We will support both modes via a flag (`--at open|close`). Default: `close` (more conservative).

### 3.7 Label set (outcomes — never used as features)
For each day `t`: `fwd_ret_1 = close[t+1]/close[t]-1`, `fwd_ret_3`, `fwd_ret_5`, plus
`fwd_maxdd_5` (worst close-to-close drawdown over t+1..t+5) and `fwd_maxup_5`.

---

## 4. The matching engine

Two complementary similarity layers; outputs are blended.

### 4.1 Snapshot k-NN (state similarity)
- Build a vector from the Section-3 features for each day.
- **Normalize** each feature to a z-score using **only data available up to `t`**
  (expanding/rolling mean & std → preserves point-in-time). Categorical features
  (`cpr_rel`, `open_vs_cpr`) are one-hot or ordinal.
- Distance = weighted Euclidean. Weights are a config dict (default: equal within group,
  groups normalized so no single group dominates). Distance metric pluggable (Euclidean,
  cosine, Mahalanobis later).
- Retrieve the `k` nearest *prior* days (default k = 50, configurable).

### 4.2 Shape DTW (trajectory similarity)
- Encode the trailing `W`-bar path (default W = 10) of a few core series
  (normalized close path, RSI path, ADX path).
- Dynamic Time Warping distance to all prior non-overlapping windows (Section 2.2 embargo).
- Retrieve top-`k` shape neighbors.

### 4.3 Blending
Each candidate day gets a combined score `α·z(snap_dist) + (1−α)·z(dtw_dist)` (default
α = 0.6). Final neighbor set = top-k by combined score. α and k are config, not magic numbers.

---

## 5. Outcome aggregation & report

For the final neighbor set, aggregate the labels:

- **P(up)** at +1/+3/+5 (share of neighbors with positive fwd return)
- **Mean / median** forward return at each horizon
- **Dispersion**: 25th/75th percentile, worst-decile drawdown (`fwd_maxdd_5`)
- **Confidence score**: a function of (neighbor count, how tight the distances are, and how
  *consistent* the outcomes are). Loud disagreement among neighbors → low confidence, and we
  say so rather than averaging it away.
- **Precedents table**: the actual dates of the top ~10 neighbors with their forward returns,
  so the analyst can pull the charts.
- **Baseline comparison**: same stats for *all* prior days (unconditional). The analog is only
  interesting insofar as it differs from the base rate.

Report format mirrors the odds-table style already in `cpr_probability.py`.

---

## 6. Reuse map (what we build on)

| Need | Reuse from | New work |
|---|---|---|
| CPR levels + 6-way relationship | `cpr_ohlc_compare.py`, `cpr_probability.py`, `nifty_cpr_relationship.py` | refactor into a callable, vectorized helper |
| RSI(14) | `patterns.py` | none |
| SMA/EMA (50/200) | `patterns.py`, `swing_scanner.py` | none |
| ATR(20) | `swing_backtest.py` | none |
| Volume features | `patterns.py` | minor adaptation for index |
| **ADX(14)** | — | **build (Section 3.3)** |
| Backtest loop pattern | `swing_backtest.py` | adapt for walk-forward eval |
| Odds-table reporting | `cpr_probability.py` | adapt |
| Data store | `prices.parquet`, `bt_nifty_daily.parquet` | extend to 15y index history |

Stack stays pure **pandas + numpy + fastparquet** (repo convention; no TA library). DTW via a
small dependency-free implementation or `dtaidistance` if we choose to add it (decision in Phase 2).

---

## 7. Module / file layout

```
nse_screener/analog/
  DESIGN.md            ← this document
  data_prep.py         ← Phase 0: pull/extend 15y Nifty & Bank Nifty daily → parquet
  features.py          ← Phase 1: vectorized feature library (+ ADX) → features_<index>.parquet
  indicators.py        ← shared indicator fns (ADX lives here; RSI/ATR/SMA imported/re-exported)
  engine.py            ← Phase 2: k-NN + DTW matcher, point-in-time guard, normalization
  report.py            ← Phase 3: outcome aggregation, odds tables, precedents, confidence
  validate.py          ← Phase 4: walk-forward edge test vs baseline, calibration
  today.py             ← Phase 5: CLI — "what does today look like?" daily report
  config.py            ← weights, k, W, α, horizons (no magic numbers scattered in code)
```

Outputs:
```
data/features_nifty.parquet, data/features_banknifty.parquet
data/analog_report_<date>.json     (machine-readable, for dashboard)
```

---

## 8. Validation methodology (Phase 4 — the credibility test)

A precedent engine is worthless if its "70% up" calls don't actually come up 70% of the time.
We test this out-of-sample:

1. **Walk-forward**: for each day in a held-out span, build the neighbor set using *only prior
   data*, record the predicted P(up) and the realized outcome.
2. **Calibration plot**: bucket predictions (50–60%, 60–70%, …) and check realized hit-rate per
   bucket. A well-calibrated engine sits on the diagonal.
3. **Edge vs baseline**: does conditioning on the analog beat the unconditional base rate
   (and a 50/50 coin)? Report lift, and whether it survives transaction costs for a naive
   "act on high-confidence calls" rule.
4. **Stability**: does edge hold across regimes (2008, 2013, 2020, 2022 drawdowns vs bull runs)?
   An engine that only works in trending markets must be labeled as such.

**We publish the failures.** If calibration is poor or edge is within noise, that goes in the
report verbatim. The tool's value is honest probabilities, not flattering ones.

### 8.1 Phase 4 RESULTS — DIRECTION (Nifty, +5d) — FAIL (2026-06-27)
Walk-forward 2018-01-02 → 2026-06, horizon +5d. Unconditional up-rate (base) ≈ 56%.

| Config | queries | Brier skill vs base | hi-conf up-call rate | mean fwd5 (hi-conf) |
|---|---|---|---|---|
| Snapshot-only (step 2) | 1042 | **−4.1%** | 53.1% (base 56.7%) | −0.08% |
| Blend +DTW (step 5) | 417 | **−2.7%** | 54.8% (base 55.6%) | +0.18% |

- **Calibration is flat-to-inverted.** Mid buckets (0.50–0.65) are roughly on the diagonal,
  but the top bucket (predicted ≥70% up) realized only **45%** (snapshot) / **59%** (blend) —
  i.e. when analog agreement is *highest*, forward direction is *not* higher, sometimes lower
  (over-extension / mean-reversion at extremes).
- **No directional edge.** High-confidence "up" calls do not beat the ~56% base rate; the
  ~+0.22% mean fwd5 across all queries is just the market's upward drift, not skill.
- **Interpretation.** 1–5 day *index direction* is close to a drift-plus-noise random walk;
  this feature set + analog approach does not forecast it. Consistent with market efficiency
  for short-horizon index timing. **Do not ship a directional "prediction."**

### 8.2 Implication / next research directions (unshipped)
The engine machinery is sound (PIT-safe, validated indicators, calibrated *harness*). The
*target* is the problem. Candidate pivots, each to be re-run through validate.py before any ship:
1. **Predict the DISTRIBUTION, not the direction** — forward *range / realized volatility /
   max drawdown*. Volatility clusters, so analogs of high-ATR days plausibly forecast high-ATR
   days. This is desk-useful (option pricing, risk, position sizing) and the honest strength of
   an analog engine. **Recommended next test.**
2. **Cross-section over timing** — apply the same engine to single stocks (relative outcomes),
   where there is more exploitable structure than in index timing.
3. **Regime conditioning** — only emit a call when confidence is high AND the analog set is
   tight; accept far lower coverage.
Direction forecasting is **shelved** pending evidence, per §8.

### 8.3 Phase 4b RESULTS — VOLATILITY / RANGE — PASS (2026-06-27)
Walk-forward 2018-01-02 → 2026-06. Target = forward 5-day high-low range (`fwd_hl_range_5`)
and annualized realized vol (`fwd_rv_5`). Baseline that matters = **persistence** (forward
range ≈ trailing range), which is strong because volatility is autocorrelated.

| Config | n | analog ρ | persistence ρ | MAE skill vs persistence | terciles (low→high) | q80 calib |
|---|---|---|---|---|---|---|
| Nifty range, snapshot | 1042 | 0.422 | 0.363 | **+17.0%** | 2.4%→4.1% | 24% |
| Nifty range, blend+DTW | 417 | 0.404 | 0.344 | **+15.1%** | 2.4%→3.9% | 24% |
| Nifty realized-vol | 1042 | 0.418 | 0.384 | **+49.6%** | 9.9%→18.6% | 22% |
| BankNifty range, snapshot | 843 | 0.526 | 0.510 | **+7.0%** | 2.9%→6.0% | 17% |

- Analog beats the persistence baseline on **both** rank-ordering and MAE in every config.
- Predicted terciles are cleanly monotonic (high-tercile realized range ≈1.6–1.9× low-tercile).
- 80th-pct band calibration lands at 17–24% (target 20%): close, slightly tight on Nifty.
- BankNifty edge over persistence is thinner (+7%) — its vol is more persistent — but ordering
  and monotonicity are the strongest of the set.
- **Conclusion: the analog engine forecasts the forward distribution (range/vol) with genuine
  out-of-sample skill beyond persistence.** This is the shippable product. Phase 5 surfaces
  range/vol context (option straddle width, expected drawdown, position sizing), NOT direction.

### Product reframe (supersedes §1's directional framing)
The shipped engine answers: *"Days like today historically ran a ~X% 5-day range (vs ~Y%
normally); expect the wider/narrower tape."* Direction is reported only as a low-confidence
context line drawn from the same neighbors, explicitly flagged as not validated (§8.1).

---

## 9. Known pitfalls & how we handle them

| Pitfall | Mitigation |
|---|---|
| **Look-ahead bias** | Section 2.1; normalization uses only past data; labels never features |
| **Window self-match / leakage** | Section 2.2 embargo |
| **Regime change** (today unlike anything in 15y) | Confidence score collapses when nearest neighbors are far; we flag "no strong precedent" instead of forcing a call |
| **Multiple testing / data dredging** | Fixed feature set decided up front (this doc); validation on held-out span only; no tuning weights to maximize backtest P&L |
| **Overlapping forward windows inflate significance** | Report effective sample size; horizons report independent-ish via spacing where it matters |
| **Index volume quality** | Flag at build; graceful fallback (Section 3.4) |
| **Survivorship / corp actions** | Indices are corp-action-clean; using adjusted series |

---

## 10. Phased roadmap

| Phase | Deliverable | Independently testable? |
|---|---|---|
| **0. Data** | 15y Nifty + Bank Nifty daily parquet, consistent schema | yes — row counts, date coverage, no gaps |
| **1. Features** | `features.py` + `indicators.py` (ADX), feature parquet | yes — spot-check ADX/RSI/CPR vs known values; no NaN leakage |
| **2. Engine** | `engine.py` k-NN + DTW with PIT guard | yes — query a known date, inspect neighbors by hand |
| **3. Report** | `report.py` odds tables + precedents + confidence | yes — output matches manual aggregation |
| **4. Validation** | `validate.py` walk-forward, calibration, edge vs baseline | yes — this *is* the test |
| **5. Daily report** | `today.py` CLI + JSON for dashboard | yes — run on today's data |

Recommended build order is exactly the above; each phase gates the next. Phase 4 is the
go/no-go: if the engine shows no calibrated edge, we either revise the feature set (documented)
or shelve it honestly rather than shipping a tool that misleads.

---

## 11. Example target output (Phase 5)

```
NIFTY ANALOG REPORT — 2026-06-27 (close basis)
State: above 50DMA (+2.1%) & 200DMA (+8.4%), RSI 61 (rising), ADX 22 (firming, +DI lead),
       volume 0.9x avg, CPR narrow (18th pctile), prior day Inside-Higher.

Nearest 50 precedents (snapshot+shape blend, mean dist 0.42 — MODERATE confidence):
  Horizon   P(up)   median   mean    25th    75th    worst-decile DD
  +1 day    62%     +0.31%   +0.28%  -0.22%  +0.74%  -0.9%
  +3 day    66%     +0.71%   +0.66%  -0.51%  +1.62%  -1.8%
  +5 day    68%     +1.02%   +0.94%  -0.83%  +2.41%  -2.4%
  (baseline +5d: 56% up, median +0.40%)  → +12pp lift over base rate

Top precedents:  2014-11-18 (+5d +1.4%), 2017-07-03 (+5d +2.1%), 2019-09-23 (+5d -0.6%), ...
Confidence: MODERATE — neighbors reasonably tight, outcomes 68% one-directional.
```

---

## 12. Open decisions (to confirm before/within each phase)

1. **DTW dependency**: hand-rolled vs `dtaidistance`. (Phase 2)
2. **Index volume**: ~~keep, fallback, or drop~~ → **RESOLVED**: volume features are
   *required*. yfinance index volume is reliable only from 2013 (Nifty) / 2011 (BankNifty),
   so the matchable candidate pool is effectively **post-2013 (~13y)**. The engine drops
   pre-2013 bars (incl. the 2008 GFC) from matching. Accepted trade-off for complete,
   comparable feature vectors over fudged/partial ones.
3. **Forecast basis default**: `close` vs `open`. (Defaulting to `close`.)
4. **History depth**: 15y proposed; extend to inception if clean data is gettable. (Phase 0)
5. **Confidence score formula**: ~~exact weighting~~ → **RESOLVED** (`engine._confidence`):
   `confidence = sample · (0.5·tightness + 0.5·consistency)`, where
   `tightness = clip((pool_median − neigh_median)/(pool_median − pool_min), 0, 1)`,
   `consistency = |2·P(up@5d) − 1|`, `sample = clip(n_realized/k, 0, 1)`. A label
   (WEAK / MIXED / LOW / MODERATE / HIGH) is always attached so a weak match is never
   silently averaged away.
