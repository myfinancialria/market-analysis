"""
Central config for the Market Analog Engine. No magic numbers scattered in code —
every tunable lives here. See DESIGN.md.
"""

# --- universe -------------------------------------------------------------
# name -> yfinance symbol. yfinance is the single authoritative source
# (longer + more current history than the Fyers bt_*.parquet files, and it
#  carries index volume from ~2013 for Nifty / ~2011 for Bank Nifty).
SYMBOLS = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
}

# --- indicator periods ----------------------------------------------------
RSI_N = 14
ADX_N = 14
ATR_N = 20
DMA_FAST = 50
DMA_SLOW = 200
SLOPE_LOOKBACK = 5          # bars for rsi/dma slope
VOL_AVG_N = 50              # volume moving-average window
HIGHLOW_N = 20             # distance-from-recent-high/low window
CPR_WIDTH_PCTILE_N = 252   # rolling window for CPR-width percentile rank

# --- forward horizons (labels) -------------------------------------------
HORIZONS = [1, 3, 5]        # trading days
DRAWDOWN_HORIZON = 5        # window for fwd_maxdd / fwd_maxup

# --- matching engine (Phase 2 — placeholders, not used in Phase 0/1) -----
K_NEIGHBORS = 50            # k for snapshot k-NN
WINDOW = 10                 # W bars for DTW shape match
BLEND_ALPHA = 0.6           # weight on snapshot vs DTW (1.0 = snapshot only)
FORECAST_BASIS = "close"    # "close" (conservative) or "open" (may use open_vs_cpr)

# Feature group weights for the weighted-Euclidean snapshot distance.
# Within a group, member features share the group weight equally; groups are
# normalized so no single group dominates.
GROUP_WEIGHTS = {
    "trend": 1.0,
    "momentum": 1.0,
    "strength": 1.0,
    "volume": 0.7,          # down-weighted: index volume is noisier / partly missing
    "volatility": 1.0,
    "cpr": 1.0,
}

# --- paths ----------------------------------------------------------------
import os

# analog/ sits directly under the repo root in this standalone repo.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def price_path(name):
    return os.path.join(DATA_DIR, f"analog_{name}_daily.parquet")


def features_path(name):
    return os.path.join(DATA_DIR, f"features_{name}.parquet")
