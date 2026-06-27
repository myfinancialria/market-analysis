# Market Analysis — Volatility & Range Outlook

A **precedent engine** for Nifty & Bank Nifty. It encodes each trading day as a feature
vector (price, volume, RSI, ADX, 50/200-DMA, CPR), finds the most-similar historical
sessions via k-NN + DTW, and reports **the distribution of forward volatility / trading
range** that followed.

🌐 **Live site:** https://myfinancialria.github.io/market-analysis/
(morning outlook + evening update, auto-refreshed on a schedule)

## The honest headline

This was built and validated end-to-end with walk-forward, point-in-time testing
(see [`analog/DESIGN.md`](analog/DESIGN.md) §8):

- **Direction (next 1–5 day up/down): FAILS out-of-sample.** Negative Brier skill,
  calibration inverts at the extremes. Short-horizon index direction is drift-plus-noise.
  **Not shipped as a signal** — shown on the site only as flagged, unvalidated context.
- **Volatility / range: PASSES.** Beats a volatility-*persistence* baseline on rank-correlation
  and MAE out-of-sample (+7% to +50% across Nifty/Bank Nifty, both targets, DTW blend),
  with monotonic prediction terciles and good quantile calibration. **This is the product.**

The site tells you *how wide* the tape is likely to be ("days like today historically ran a
~X% 5-day range vs ~Y% normally"), which is what an options/risk desk actually needs — not a
direction call it can't honestly make.

## Layout

```
analog/
  config.py         tunables (periods, k, window, weights, paths)
  data_prep.py      fetch full-history daily OHLCV (yfinance ^NSEI / ^NSEBANK)
  indicators.py     SMA/EMA/RSI/ATR/ADX/CPR (validated vs TA-Lib)
  features.py       point-in-time feature + forward-label builder
  engine.py         k-NN snapshot + DTW shape matcher (embargoed, PIT-safe)
  report.py         per-query odds tables + analog-vs-baseline lift + JSON
  validate.py       walk-forward DIRECTION test (the FAIL)
  validate_vol.py   walk-forward VOL/RANGE test (the PASS)
  report_site.py    morning/evening report generator for the website
  DESIGN.md         full methodology, correctness rules, and results
docs/               GitHub Pages site (index.html + data/*.json)
.github/workflows/  scheduled regenerate + Pages deploy
```

## Run locally

```bash
pip install -r requirements.txt
python analog/data_prep.py            # → data/analog_*_daily.parquet
python analog/features.py             # → data/features_*.parquet
python analog/report_site.py --mode morning   # → docs/data/morning_*.json
python analog/validate_vol.py --name nifty     # reproduce the validation
```

## Caveats

- yfinance index volume is reliable only from ~2013, so the matchable history is ~13y
  (the engine is blind to the 2008 GFC by design — see DESIGN.md §12.2).
- Educational / research use only. **Not investment advice.**
