"""
Shared, vectorized indicator library for the Analog Engine (DESIGN.md §3).

Pure pandas/numpy (repo convention — no TA library). RSI matches the existing
patterns.py implementation; ATR matches swing_backtest.py; CPR matches
cpr_probability.py. ADX is the one genuinely new indicator and is implemented
to Wilder's spec (see DESIGN.md §3.3).

Every function is point-in-time safe: value at bar t uses only bars <= t.
"""
import numpy as np
import pandas as pd


# --- moving averages ------------------------------------------------------
def sma(series, n):
    return series.rolling(n).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def wilder(series, n):
    """Wilder's smoothing == EMA with alpha = 1/n (RMA)."""
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


# --- RSI (matches patterns.py) -------------------------------------------
def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# --- True Range / ATR (matches swing_backtest.py) ------------------------
def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high, low, close, n=20):
    return true_range(high, low, close).rolling(n).mean()


# --- ADX / DI (NEW — Wilder, DESIGN.md §3.3) -----------------------------
def adx(high, low, close, n=14):
    """
    Returns DataFrame with columns adx, plus_di, minus_di. Wilder-smoothed.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_w = wilder(tr, n)

    plus_di = 100 * wilder(plus_dm, n) / atr_w
    minus_di = 100 * wilder(minus_dm, n) / atr_w

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_line = wilder(dx, n)

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


# --- CPR (matches cpr_probability.py) ------------------------------------
def cpr_levels(high, low, close):
    """
    Vectorized Central Pivot Range from a bar's own OHLC.
    Returns DataFrame: pivot (P), top (BC/TC upper), bot (lower).
    To get the CPR *valid for day t*, feed day t-1's H/L/C (caller shifts).
    """
    p = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2 * p - bc
    top = pd.concat([tc, bc], axis=1).max(axis=1)
    bot = pd.concat([tc, bc], axis=1).min(axis=1)
    return pd.DataFrame({"pivot": p, "top": top, "bot": bot})


# Two-day CPR relationship taxonomy (matches nifty_cpr_relationship.py /
# cpr_probability.relationship). Returned as a small integer code so it can be
# one-hot encoded by the feature layer.
CPR_REL = {
    "higher": 0, "lower": 1, "inside": 2, "outside": 3,
    "overlap_higher": 4, "overlap_lower": 5,
}


def cpr_relationship_code(t_top, t_bot, t_piv, y_top, y_bot, y_piv):
    """Classify today's CPR (t) vs yesterday's (y). Scalar inputs -> code int."""
    if t_bot > y_top:
        return CPR_REL["higher"]
    if t_top < y_bot:
        return CPR_REL["lower"]
    if t_bot >= y_bot and t_top <= y_top:
        return CPR_REL["inside"]
    if t_bot <= y_bot and t_top >= y_top:
        return CPR_REL["outside"]
    return CPR_REL["overlap_higher"] if t_piv > y_piv else CPR_REL["overlap_lower"]


def cpr_relationship_series(cpr_df):
    """
    Vectorized-ish two-day relationship over a cpr_levels() frame indexed by bar.
    Row t compares CPR(t) vs CPR(t-1). First row -> NaN.
    """
    top, bot, piv = cpr_df["top"], cpr_df["bot"], cpr_df["pivot"]
    yt, yb, yp = top.shift(1), bot.shift(1), piv.shift(1)
    codes = []
    for i in range(len(cpr_df)):
        if i == 0 or pd.isna(yt.iloc[i]):
            codes.append(np.nan)
        else:
            codes.append(cpr_relationship_code(
                top.iloc[i], bot.iloc[i], piv.iloc[i],
                yt.iloc[i], yb.iloc[i], yp.iloc[i]))
    return pd.Series(codes, index=cpr_df.index, dtype="float64")
