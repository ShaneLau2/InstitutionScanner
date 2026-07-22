"""
indicators.py — Technical indicator computation engine.

All indicator functions accept a pandas DataFrame with at minimum
['Open', 'High', 'Low', 'Close', 'Volume'] columns and return
either a Series, a DataFrame, or mutate the input in-place.

Design principles:
- Vectorised (numpy / pandas) where possible — avoid Python-level loops.
- Every public function returns None (mutates df) or a Series, not a mix.
- Functions are independent and idempotent — call order doesn't matter.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

# Suppress harmless numpy warnings (log of zero, divide by zero in R², etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import (
    AD_SLOPE_PERIOD,
    ADX_PERIOD,
    ATR_PERIODS,
    BB_PERIOD,
    BB_STD,
    CCI_PERIOD,
    CMF_PERIOD,
    DONCHIAN_PERIOD,
    EMA_PERIODS,
    ENABLE_ADVANCED_INDICATORS,
    ENABLE_VOLUME_PROFILE,
    HV_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MA_PERIODS,
    MFI_PERIOD,
    OBV_SLOPE_PERIOD,
    REGRESSION_PERIOD,
    ROC_PERIOD,
    RSI_PERIODS,
    VOLUME_MA_PERIODS,
    VOLUME_PROFILE_BINS,
    VOLUME_PROFILE_LOOKBACK,
    VOLUME_RATIO_PERIODS,
    VOLUME_TREND_PERIOD,
    VOLUME_ZSCORE_PERIOD,
    VWAP_PERIOD,
)

logger = logging.getLogger("scanner_gui.indicators")


# ======================================================================
# Helpers
# ======================================================================

def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, returning *fill* where denominator is 0 or NaN."""
    result = a / b.replace(0, np.nan)
    return result.fillna(fill)


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series to a plain NumPy float array."""
    return np.asarray(series.astype(np.float64), dtype=np.float64)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear regression slope over right-aligned windows.

    Vectorized O(n) implementation using cumulative sums.
    For relative positions x = [0, 1, ..., n-1]:
      slope = (Σ(x*y) - n·x̄·ȳ) / Σ((x-x̄)²)
    """
    y = series.astype(np.float64)
    n = window
    length = len(y)

    if length < n:
        return pd.Series(np.full(length, np.nan), index=series.index)

    # Precompute constants for x = [0, 1, ..., n-1]
    x_mean = (n - 1) / 2.0
    denom = n * (n - 1) * (n + 1) / 12.0  # Σ((x-x̄)²)
    min_periods = max(2, n // 2)

    # Fill NaN for cumsum, but track actual valid count
    y_filled = y.fillna(0)
    idx = np.arange(length)

    # Cumulative sums
    cum_y = y_filled.cumsum()
    cum_jy = (y_filled * idx).cumsum()

    # Rolling sums via cumulative difference
    prev_cum_y = cum_y.shift(n).fillna(0)
    prev_cum_jy = cum_jy.shift(n).fillna(0)

    sum_y = cum_y - prev_cum_y            # Σ(y) in window
    sum_jy = cum_jy - prev_cum_jy          # Σ(j·y[j]) with absolute j
    start_idx = pd.Series(idx - n + 1, index=y.index)
    sum_rel_y = sum_jy - start_idx * sum_y  # Σ(x_rel·y) with relative x

    y_mean = sum_y / n
    slope = (sum_rel_y - n * x_mean * y_mean) / denom

    # Apply min_periods
    valid_count = y.rolling(n, min_periods=1).count()
    slope[valid_count < min_periods] = np.nan
    slope.iloc[:n - 1] = np.nan

    return slope


def _rolling_r2(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling R² over right-aligned windows.

    Vectorized O(n) — shares cumsum with _rolling_slope.
    r² = (Σ((x-x̄)(y-ȳ)))² / (Σ((x-x̄)²) · Σ((y-ȳ)²))
    """
    y = series.astype(np.float64)
    n = window
    length = len(y)

    if length < n:
        return pd.Series(np.full(length, np.nan), index=series.index)

    # Precompute constants
    x_mean = (n - 1) / 2.0
    denom_x = n * (n - 1) * (n + 1) / 12.0  # Σ((x-x̄)²)
    min_periods = max(2, n // 2)

    y_filled = y.fillna(0)
    idx = np.arange(length)

    # Three cumulative sums: y, j·y, y²
    cum_y = y_filled.cumsum()
    cum_jy = (y_filled * idx).cumsum()
    cum_y2 = (y_filled ** 2).cumsum()

    prev_cum_y = cum_y.shift(n).fillna(0)
    prev_cum_jy = cum_jy.shift(n).fillna(0)
    prev_cum_y2 = cum_y2.shift(n).fillna(0)

    sum_y = cum_y - prev_cum_y
    sum_jy = cum_jy - prev_cum_jy
    sum_y2 = cum_y2 - prev_cum_y2

    start_idx = pd.Series(idx - n + 1, index=y.index)
    sum_rel_y = sum_jy - start_idx * sum_y  # Σ(x_rel·y)

    y_mean = sum_y / n
    numerator = sum_rel_y - n * x_mean * y_mean  # Σ((x-x̄)(y-ȳ))

    y_var = sum_y2 - sum_y * y_mean  # Σ((y-ȳ)²)

    result = pd.Series(np.full(length, np.nan), index=y.index)
    mask = (denom_x > 0) & (y_var > 0)
    result[mask] = (numerator[mask] ** 2) / (denom_x * y_var[mask])

    # Apply min_periods
    valid_count = y.rolling(n, min_periods=1).count()
    result[valid_count < min_periods] = np.nan
    result.iloc[:n - 1] = np.nan

    return result


# ======================================================================
# Price Indicators
# ======================================================================

def compute_moving_averages(df: pd.DataFrame) -> None:
    close = df["Close"]
    for period in MA_PERIODS:
        df[f"MA{period}"] = close.rolling(window=period, min_periods=period // 2).mean()


def compute_ema(df: pd.DataFrame) -> None:
    close = df["Close"]
    for period in EMA_PERIODS:
        df[f"EMA{period}"] = close.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame) -> None:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["_TR"] = true_range  # cached for compute_adx
    for period in ATR_PERIODS:
        df[f"ATR{period}"] = true_range.rolling(window=period, min_periods=period // 2).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> None:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    cond_plus = (up_move > down_move) & (up_move > 0)
    cond_minus = (down_move > up_move) & (down_move > 0)
    plus_dm[cond_plus] = up_move[cond_plus]
    minus_dm[cond_minus] = down_move[cond_minus]
    # Use cached true_range from compute_atr if available
    if "_TR" in df.columns:
        tr = df["_TR"]
    else:
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period, min_periods=period // 2).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period // 2).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period // 2).mean() / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(window=period, min_periods=period // 2).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di


def compute_cci(df: pd.DataFrame, period: int = CCI_PERIOD) -> None:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=period // 2).mean()
    # Vectorized MAD using numpy sliding_window_view (no Python lambda)
    arr = tp.values.astype(np.float64)
    n = period
    mad = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(arr, n)
        w_means = windows.mean(axis=1, keepdims=True)
        mad[n - 1:] = np.mean(np.abs(windows - w_means), axis=1)
    df["CCI"] = (tp - sma_tp) / (0.015 * pd.Series(mad, index=df.index))


def compute_roc(df: pd.DataFrame, period: int = ROC_PERIOD) -> None:
    close = df["Close"]
    df[f"ROC{period}"] = ((close - close.shift(period)) / close.shift(period).replace(0, np.nan)) * 100


def compute_52week_levels(df: pd.DataFrame) -> None:
    close = df["Close"]
    window = 252
    high_52w = close.rolling(window=window, min_periods=window // 2).max()
    low_52w = close.rolling(window=window, min_periods=window // 2).min()
    df["High52W"] = high_52w
    df["Low52W"] = low_52w
    df["DistToHigh52W"] = ((close - high_52w) / high_52w.replace(0, np.nan)) * 100
    df["DistToLow52W"] = ((close - low_52w) / low_52w.replace(0, np.nan)) * 100


# ======================================================================
# Volume Indicators
# ======================================================================

def compute_volume_mas(df: pd.DataFrame) -> None:
    vol = df["Volume"]
    for period in VOLUME_MA_PERIODS:
        df[f"VolMA{period}"] = vol.rolling(window=period, min_periods=period // 2).mean()


def compute_volume_ratios(df: pd.DataFrame) -> None:
    vol = df["Volume"]
    for period in VOLUME_RATIO_PERIODS:
        avg = vol.rolling(window=period, min_periods=period // 2).mean()
        df[f"VolRatio{period}"] = _safe_divide(vol, avg)


def compute_relative_volume(df: pd.DataFrame) -> None:
    vol = df["Volume"]
    avg_5 = vol.rolling(window=5, min_periods=3).mean()
    df["RelVolume"] = _safe_divide(vol, avg_5)


def compute_volume_zscore(df: pd.DataFrame, period: int = VOLUME_ZSCORE_PERIOD) -> None:
    vol = df["Volume"]
    roll_mean = vol.rolling(window=period, min_periods=period // 2).mean()
    roll_std = vol.rolling(window=period, min_periods=period // 2).std()
    df["VolZScore"] = _safe_divide(vol - roll_mean, roll_std)


def compute_volume_trend(df: pd.DataFrame, period: int = VOLUME_TREND_PERIOD) -> None:
    df["VolTrend"] = _rolling_slope(df["Volume"], period)


# ======================================================================
# Money Flow Indicators
# ======================================================================

def compute_obv(df: pd.DataFrame) -> None:
    close, vol = df["Close"], df["Volume"]
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    direction[0] = 0
    df["OBV"] = (vol * direction).cumsum()


def compute_obv_slope(df: pd.DataFrame, period: int = OBV_SLOPE_PERIOD) -> None:
    if "OBV" not in df.columns:
        compute_obv(df)
    df["OBV_Slope"] = _rolling_slope(df["OBV"], period)


def compute_ad_line(df: pd.DataFrame) -> None:
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    hl_range = high - low
    clv = pd.Series(0.0, index=df.index)
    mask = hl_range > 0
    clv[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / hl_range[mask]
    df["AD"] = (clv * vol).cumsum()


def compute_ad_slope(df: pd.DataFrame, period: int = AD_SLOPE_PERIOD) -> None:
    if "AD" not in df.columns:
        compute_ad_line(df)
    df["AD_Slope"] = _rolling_slope(df["AD"], period)


def compute_cmf(df: pd.DataFrame, period: int = CMF_PERIOD) -> None:
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    hl_range = high - low
    mf_multiplier = pd.Series(0.0, index=df.index)
    mask = hl_range > 0
    mf_multiplier[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / hl_range[mask]
    money_flow_volume = mf_multiplier * vol
    df["CMF"] = money_flow_volume.rolling(window=period, min_periods=period // 2).sum() / \
                vol.rolling(window=period, min_periods=period // 2).sum()


def compute_mfi(df: pd.DataFrame, period: int = MFI_PERIOD) -> None:
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    tp = (high + low + close) / 3
    raw_money_flow = tp * vol
    pos_flow = pd.Series(0.0, index=df.index)
    neg_flow = pd.Series(0.0, index=df.index)
    up = tp > tp.shift(1)
    down = tp < tp.shift(1)
    pos_flow[up] = raw_money_flow[up]
    neg_flow[down] = raw_money_flow[down]
    pos_sum = pos_flow.rolling(window=period, min_periods=period // 2).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period // 2).sum()
    mfi = pd.Series(50.0, index=df.index)
    positive_only = (neg_sum == 0) & (pos_sum > 0)
    negative_only = (pos_sum == 0) & (neg_sum > 0)
    both_nonzero = (pos_sum > 0) & (neg_sum > 0)
    mfi[positive_only] = 100.0
    mfi[negative_only] = 0.0
    money_ratio = pos_sum[both_nonzero] / neg_sum[both_nonzero]
    mfi[both_nonzero] = 100 - (100 / (1 + money_ratio))
    df["MFI"] = mfi


def compute_vwap(df: pd.DataFrame) -> None:
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    tp = (high + low + close) / 3
    cum_pv = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    df["VWAP"] = _safe_divide(cum_pv, cum_vol)
    df["VWAP_Dist"] = ((close - df["VWAP"]) / df["VWAP"].replace(0, np.nan)) * 100


# ======================================================================
# Trend Indicators
# ======================================================================

def compute_regression(df: pd.DataFrame, period: int = REGRESSION_PERIOD) -> None:
    close = df["Close"]
    df["RegSlope"] = _rolling_slope(close, period)
    df["RegR2"] = _rolling_r2(close, period)


def compute_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> None:
    close = df["Close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = (ema_fast - ema_slow).ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]


def compute_rsi(df: pd.DataFrame) -> None:
    close = df["Close"]
    delta = close.diff()
    for period in RSI_PERIODS:
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rsi = pd.Series(50.0, index=df.index)
        positive_only = (avg_loss == 0) & (avg_gain > 0)
        negative_only = (avg_gain == 0) & (avg_loss > 0)
        both_nonzero = (avg_gain > 0) & (avg_loss > 0)
        rsi[positive_only] = 100.0
        rsi[negative_only] = 0.0
        rs = avg_gain[both_nonzero] / avg_loss[both_nonzero]
        rsi[both_nonzero] = 100 - (100 / (1 + rs))
        df[f"RSI{period}"] = rsi


# ======================================================================
# Volatility Indicators
# ======================================================================

def compute_historical_volatility(df: pd.DataFrame) -> None:
    close = df["Close"].astype(np.float64)
    log_ret = pd.Series(np.log(close / close.shift(1)), index=df.index, dtype=np.float64)
    for period in HV_PERIODS:
        roll_std = log_ret.rolling(window=period, min_periods=period // 2).std()
        df[f"HV{period}"] = roll_std * np.sqrt(252) * 100


def compute_atr_compression(df: pd.DataFrame, period: int | None = None) -> None:
    if period is None:
        period = ATR_PERIODS[1]
    if f"ATR{ATR_PERIODS[0]}" not in df.columns:
        compute_atr(df)
    short = df.get(f"ATR{ATR_PERIODS[0]}")
    long = df.get(f"ATR{period}")
    if short is not None and long is not None:
        df["ATR_Compression"] = _safe_divide(short, long, fill=1.0)


def compute_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD) -> None:
    close = df["Close"]
    middle = close.rolling(window=period, min_periods=period // 2).mean()
    roll_std = close.rolling(window=period, min_periods=period // 2).std()
    df["BB_Middle"] = middle
    df["BB_Upper"] = middle + std * roll_std
    df["BB_Lower"] = middle - std * roll_std
    df["BB_Width"] = _safe_divide(df["BB_Upper"] - df["BB_Lower"], middle) * 100
    df["BB_Position"] = _safe_divide(close - df["BB_Lower"], df["BB_Upper"] - df["BB_Lower"])


def compute_donchian(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> None:
    high, low = df["High"], df["Low"]
    upper = high.rolling(window=period, min_periods=period // 2).max()
    lower = low.rolling(window=period, min_periods=period // 2).min()
    df["DC_Upper"] = upper
    df["DC_Lower"] = lower
    df["DC_Middle"] = (upper + lower) / 2
    df["DC_Width"] = _safe_divide(upper - lower, df["DC_Middle"]) * 100


# ======================================================================
# Volume Profile (HVN / LVN)
# ======================================================================

def compute_volume_profile(df: pd.DataFrame, bins: int = VOLUME_PROFILE_BINS, lookback: int = VOLUME_PROFILE_LOOKBACK) -> None:
    if len(df) < lookback:
        lookback = len(df)
    subset = df.iloc[-lookback:]
    close, high, low, vol = subset["Close"], subset["High"], subset["Low"], subset["Volume"]
    price_min, price_max = low.min(), high.max()
    if price_min == price_max:
        df["VP_HVN_Center"] = price_min
        df["DistToHVN_Pct"] = 0.0
        return

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (bins,)

    # Determine which bins each row spans — fully vectorized
    lo_idx = np.clip(np.digitize(low.values, bin_edges) - 1, 0, bins - 1)
    hi_idx = np.clip(np.digitize(high.values, bin_edges) - 1, 0, bins - 1)
    hi_idx = np.maximum(hi_idx, lo_idx)  # ensure at least 1 bin per row
    n_spanned = hi_idx - lo_idx + 1      # (lookback,)

    # Build coverage mask via broadcasting: (lookback, bins)
    cover_mask = (
        (low.values[:, None] <= bin_centers[None, :]) &
        (high.values[:, None] >= bin_centers[None, :])
    )

    # Weight per row: volume divided evenly across spanned bins
    weights = vol.values.astype(np.float64) / n_spanned.astype(np.float64)
    profile = (cover_mask * weights[:, None]).sum(axis=0)  # (bins,)

    if profile.sum() == 0:
        return
    threshold_hvn = np.percentile(profile[profile > 0], 67) if (profile > 0).any() else 0
    threshold_lvn = np.percentile(profile[profile > 0], 33) if (profile > 0).any() else 0
    hvn_mask = profile >= threshold_hvn
    lvn_mask = (profile > 0) & (profile <= threshold_lvn)
    current_price = close.iloc[-1]
    if hvn_mask.any():
        weighted_hvn = np.average(bin_centers[hvn_mask], weights=profile[hvn_mask])
        df["VP_HVN_Center"] = weighted_hvn
        df["DistToHVN_Pct"] = ((current_price - weighted_hvn) / weighted_hvn) * 100
        df["Above_HVN"] = current_price > weighted_hvn
    else:
        df["VP_HVN_Center"] = np.nan
        df["DistToHVN_Pct"] = np.nan
        df["Above_HVN"] = np.nan
    if lvn_mask.any():
        weighted_lvn = np.average(bin_centers[lvn_mask], weights=profile[lvn_mask])
        df["VP_LVN_Center"] = weighted_lvn
        df["DistToLVN_Pct"] = ((current_price - weighted_lvn) / weighted_lvn) * 100
    else:
        df["VP_LVN_Center"] = np.nan
        df["DistToLVN_Pct"] = np.nan


# ======================================================================
# Wyckoff Phase Detection
# ======================================================================

def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    close, vol = df["Close"], df["Volume"]
    if len(df) < 200:
        df["WyckoffPhase"] = "Unknown"
        return
    price_now = close.iloc[-1]
    # Reuse MA200 from compute_moving_averages if available
    if "MA200" in df.columns:
        ma200 = df["MA200"].iloc[-1]
        ma200_series = df["MA200"]
    else:
        ma200 = close.rolling(200, min_periods=100).mean().iloc[-1]
        ma200_series = close.rolling(200, min_periods=100).mean()
    ma200_slope = _rolling_slope(ma200_series, 60).iloc[-1]
    vol_ma20 = vol.rolling(20, min_periods=10).mean().iloc[-1]
    vol_ma60 = vol.rolling(60, min_periods=30).mean().iloc[-1]
    vol_spike = vol_ma20 > vol_ma60 * 1.5
    if "ATR14" not in df.columns:
        compute_atr(df)
    atr14 = df["ATR14"].iloc[-1]
    atr50 = df.get("ATR50", pd.Series(np.nan, index=df.index))
    atr50_now = atr50.iloc[-1] if len(atr50) > 0 else atr14
    atr_contracting = atr14 < atr50_now * 0.85
    high_60d = close.iloc[-60:].max()
    low_60d = close.iloc[-60:].min()
    dist_from_low = (price_now - low_60d) / low_60d * 100 if low_60d > 0 else 0
    dist_from_high = (high_60d - price_now) / high_60d * 100 if high_60d > 0 else 0
    high_52w = df.get("High52W", pd.Series(np.nan, index=df.index))
    low_52w = df.get("Low52W", pd.Series(np.nan, index=df.index))
    h52 = high_52w.iloc[-1] if len(high_52w) > 0 else high_60d
    l52 = low_52w.iloc[-1] if len(low_52w) > 0 else low_60d
    near_52w_low = (price_now - l52) / l52 * 100 < 5 if l52 > 0 else False
    near_52w_high = (h52 - price_now) / h52 * 100 < 5 if h52 > 0 else False
    hv20 = df.get("HV20", pd.Series(np.nan, index=df.index)).iloc[-1] if "HV20" in df.columns else np.nan
    hv60 = df.get("HV60", pd.Series(np.nan, index=df.index)).iloc[-1] if "HV60" in df.columns else np.nan
    vol_contracting = (not np.isnan(hv20) and not np.isnan(hv60) and hv20 < hv60 * 0.85)
    phase = "Unknown"
    if near_52w_low and vol_spike:
        recent_decline = (close.iloc[-20] - low_60d) / close.iloc[-20] * 100 > 15 if close.iloc[-20] > 0 else False
        if recent_decline:
            phase = "Selling Climax"
    if phase == "Unknown" and dist_from_low > 5 and dist_from_high > 15:
        recent_lowest = close.iloc[-60:].idxmin()
        if recent_lowest is not None:
            recent_pos = df.index.get_loc(recent_lowest)
            if isinstance(recent_pos, slice):
                recent_pos = recent_pos.start if recent_pos.start is not None else 0
            elif isinstance(recent_pos, np.ndarray):
                recent_pos = int(recent_pos[0]) if len(recent_pos) > 0 else 0
            else:
                recent_pos = int(recent_pos)
            bars_from_low = len(df) - recent_pos - 1
            if 5 <= bars_from_low <= 30:
                phase = "Automatic Rally"
    if phase == "Unknown" and dist_from_low < 10 and not vol_spike and atr_contracting:
        recent_high = close.iloc[-60:-10].max() if len(df) >= 70 else close.iloc[-60:].max()
        rallied = (recent_high - low_60d) / low_60d * 100 > 10 if low_60d > 0 else False
        if rallied:
            phase = "Secondary Test"
    if phase == "Unknown":
        if (price_now < ma200 and ma200_slope <= 0 and vol_ma20 > vol_ma60
                and atr_contracting and vol_contracting and dist_from_low < 20 and not near_52w_high):
            phase = "Accumulation"
    if phase == "Unknown":
        if price_now > ma200 and ma200_slope > 0 and dist_from_high < 20:
            phase = "Markup"
    if phase == "Unknown":
        if near_52w_high and vol_ma20 > vol_ma60 and not atr_contracting and ma200_slope < 0.5:
            phase = "Distribution"
    if phase == "Unknown":
        if price_now < ma200 and ma200_slope < 0 and dist_from_low > 30 and not atr_contracting:
            phase = "Markdown"
    df["WyckoffPhase"] = phase


# ======================================================================
# Master function
# ======================================================================

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    compute_moving_averages(df)
    compute_ema(df)
    compute_atr(df)
    compute_adx(df)
    compute_cci(df)
    compute_roc(df)
    compute_52week_levels(df)
    compute_volume_mas(df)
    compute_volume_ratios(df)
    compute_relative_volume(df)
    compute_volume_zscore(df)
    compute_volume_trend(df)
    compute_obv(df)
    compute_obv_slope(df)
    compute_ad_line(df)
    compute_ad_slope(df)
    compute_cmf(df)
    compute_mfi(df)
    if ENABLE_ADVANCED_INDICATORS:
        compute_vwap(df)
        compute_cci(df)
        compute_donchian(df)
        compute_volume_ratios(df)
    compute_regression(df)
    compute_macd(df)
    compute_rsi(df)
    compute_historical_volatility(df)
    compute_atr_compression(df)
    compute_bollinger_bands(df)
    if ENABLE_VOLUME_PROFILE:
        try:
            compute_volume_profile(df)
        except Exception:
            logger.debug("Volume Profile failed — skipping.", exc_info=True)
    try:
        detect_wyckoff_phase(df)
    except Exception:
        logger.debug("Wyckoff detection failed — skipping.", exc_info=True)
    return df
