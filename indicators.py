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
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

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

logger = logging.getLogger("institution_scanner.indicators")


# ======================================================================
# Helpers
# ======================================================================

def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, returning *fill* where denominator is 0 or NaN."""
    result = a / b.replace(0, np.nan)
    return result.fillna(fill)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear regression slope over *window* periods.
    Uses scipy.stats.linregress in a rolling apply (vectorised where possible).
    """
    def _slope(y: np.ndarray) -> float:
        if len(y) < 2 or np.isnan(y).all():
            return np.nan
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        return slope

    return series.rolling(window, min_periods=max(2, window // 2)).apply(_slope, raw=True)


def _rolling_r2(series: pd.Series, window: int) -> pd.Series:
    """Rolling R² (coefficient of determination) over *window* periods."""
    def _r2(y: np.ndarray) -> float:
        if len(y) < 2 or np.isnan(y).all():
            return np.nan
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope, intercept, r_value, _, _ = stats.linregress(x[mask], y[mask])
        return r_value ** 2

    return series.rolling(window, min_periods=max(2, window // 2)).apply(_r2, raw=True)


# ======================================================================
# Price Indicators
# ======================================================================

def compute_moving_averages(df: pd.DataFrame) -> None:
    """Compute SMA for all configured periods."""
    close = df["Close"]
    for period in MA_PERIODS:
        df[f"MA{period}"] = close.rolling(window=period, min_periods=period // 2).mean()


def compute_ema(df: pd.DataFrame) -> None:
    """Compute EMA for all configured periods."""
    close = df["Close"]
    for period in EMA_PERIODS:
        df[f"EMA{period}"] = close.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame) -> None:
    """Compute Average True Range for configured periods."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in ATR_PERIODS:
        df[f"ATR{period}"] = true_range.rolling(window=period, min_periods=period // 2).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> None:
    """Compute ADX (Average Directional Index)."""
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

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(window=period, min_periods=period // 2).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period // 2).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period // 2).mean() / atr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(window=period, min_periods=period // 2).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di


def compute_cci(df: pd.DataFrame, period: int = CCI_PERIOD) -> None:
    """Compute Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=period // 2).mean()
    mad = tp.rolling(window=period, min_periods=period // 2).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df["CCI"] = (tp - sma_tp) / (0.015 * mad)


def compute_roc(df: pd.DataFrame, period: int = ROC_PERIOD) -> None:
    """Compute Rate of Change."""
    close = df["Close"]
    df["ROC"] = ((close - close.shift(period)) / close.shift(period).replace(0, np.nan)) * 100


def compute_52week_levels(df: pd.DataFrame) -> None:
    """Compute rolling 52-week high / low and distance percentages."""
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
    """Compute volume moving averages for configured periods."""
    vol = df["Volume"]
    for period in VOLUME_MA_PERIODS:
        df[f"VolMA{period}"] = vol.rolling(window=period, min_periods=period // 2).mean()


def compute_volume_ratios(df: pd.DataFrame) -> None:
    """Compute Volume Ratio: current volume / average over a window."""
    vol = df["Volume"]
    for period in VOLUME_RATIO_PERIODS:
        avg = vol.rolling(window=period, min_periods=period // 2).mean()
        df[f"VolRatio{period}"] = _safe_divide(vol, avg)


def compute_relative_volume(df: pd.DataFrame) -> None:
    """
    Relative Volume: today's volume / average volume at the same
    intra-period time (for daily data this compares to the 5-day average).
    """
    vol = df["Volume"]
    avg_5 = vol.rolling(window=5, min_periods=3).mean()
    df["RelVolume"] = _safe_divide(vol, avg_5)


def compute_volume_zscore(df: pd.DataFrame, period: int = VOLUME_ZSCORE_PERIOD) -> None:
    """Z-score of volume over a rolling window."""
    vol = df["Volume"]
    roll_mean = vol.rolling(window=period, min_periods=period // 2).mean()
    roll_std = vol.rolling(window=period, min_periods=period // 2).std()
    df["VolZScore"] = _safe_divide(vol - roll_mean, roll_std)


def compute_volume_trend(df: pd.DataFrame, period: int = VOLUME_TREND_PERIOD) -> None:
    """Linear regression slope of volume over *period* days."""
    df["VolTrend"] = _rolling_slope(df["Volume"], period)


# ======================================================================
# Money Flow Indicators
# ======================================================================

def compute_obv(df: pd.DataFrame) -> None:
    """On-Balance Volume."""
    close, vol = df["Close"], df["Volume"]
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    direction[0] = 0
    obv = (vol * direction).cumsum()
    df["OBV"] = obv


def compute_obv_slope(df: pd.DataFrame, period: int = OBV_SLOPE_PERIOD) -> None:
    """Rolling slope of OBV."""
    if "OBV" not in df.columns:
        compute_obv(df)
    df["OBV_Slope"] = _rolling_slope(df["OBV"], period)


def compute_ad_line(df: pd.DataFrame) -> None:
    """Accumulation / Distribution Line (A/D)."""
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]

    hl_range = high - low
    clv = pd.Series(0.0, index=df.index)
    mask = hl_range > 0
    clv[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / hl_range[mask]

    money_flow_volume = clv * vol
    df["AD"] = money_flow_volume.cumsum()


def compute_ad_slope(df: pd.DataFrame, period: int = AD_SLOPE_PERIOD) -> None:
    """Rolling slope of A/D line."""
    if "AD" not in df.columns:
        compute_ad_line(df)
    df["AD_Slope"] = _rolling_slope(df["AD"], period)


def compute_cmf(df: pd.DataFrame, period: int = CMF_PERIOD) -> None:
    """Chaikin Money Flow."""
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]

    hl_range = high - low
    mf_multiplier = pd.Series(0.0, index=df.index)
    mask = hl_range > 0
    mf_multiplier[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / hl_range[mask]

    money_flow_volume = mf_multiplier * vol
    df["CMF"] = money_flow_volume.rolling(window=period, min_periods=period // 2).sum() / \
                vol.rolling(window=period, min_periods=period // 2).sum()


def compute_mfi(df: pd.DataFrame, period: int = MFI_PERIOD) -> None:
    """Money Flow Index."""
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

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    df["MFI"] = 100 - (100 / (1 + money_ratio))


def compute_vwap(df: pd.DataFrame) -> None:
    """
    Cumulative VWAP (Volume-Weighted Average Price).
    For daily data this is a running approximation.
    """
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
    """Linear regression slope and R² on closing price."""
    close = df["Close"]
    df["RegSlope"] = _rolling_slope(close, period)
    df["RegR2"] = _rolling_r2(close, period)


def compute_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> None:
    """MACD line, signal line, and histogram."""
    close = df["Close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = macd_line - signal_line


def compute_rsi(df: pd.DataFrame) -> None:
    """Compute RSI for all configured periods via Wilder's smoothing."""
    close = df["Close"]
    delta = close.diff()

    for period in RSI_PERIODS:
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI{period}"] = 100 - (100 / (1 + rs))


# ======================================================================
# Volatility Indicators
# ======================================================================

def compute_historical_volatility(df: pd.DataFrame) -> None:
    """Annualised historical volatility (log returns)."""
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))
    for period in HV_PERIODS:
        roll_std = log_ret.rolling(window=period, min_periods=period // 2).std()
        df[f"HV{period}"] = roll_std * np.sqrt(252) * 100  # annualised %


def compute_atr_compression(df: pd.DataFrame, period: int | None = None) -> None:
    """
    ATR Compression: ratio of short-term ATR to long-term ATR.
    Values < 1 indicate compression.
    """
    if period is None:
        period = ATR_PERIODS[1]  # default ATR50
    if f"ATR{ATR_PERIODS[0]}" not in df.columns:
        compute_atr(df)
    short = df.get(f"ATR{ATR_PERIODS[0]}")
    long = df.get(f"ATR{period}")
    if short is not None and long is not None:
        df["ATR_Compression"] = _safe_divide(short, long, fill=1.0)


def compute_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD) -> None:
    """Bollinger Bands and Bandwidth."""
    close = df["Close"]
    middle = close.rolling(window=period, min_periods=period // 2).mean()
    roll_std = close.rolling(window=period, min_periods=period // 2).std()
    upper = middle + std * roll_std
    lower = middle - std * roll_std

    df["BB_Middle"] = middle
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = _safe_divide(upper - lower, middle) * 100  # %
    df["BB_Position"] = _safe_divide(close - lower, upper - lower)  # 0–1


def compute_donchian(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> None:
    """Donchian Channels and Width."""
    high, low = df["High"], df["Low"]
    upper = high.rolling(window=period, min_periods=period // 2).max()
    lower = low.rolling(window=period, min_periods=period // 2).min()
    middle = (upper + lower) / 2

    df["DC_Upper"] = upper
    df["DC_Lower"] = lower
    df["DC_Middle"] = middle
    df["DC_Width"] = _safe_divide(upper - lower, middle) * 100


# ======================================================================
# Volume Profile (HVN / LVN)
# ======================================================================

def compute_volume_profile(
    df: pd.DataFrame,
    bins: int = VOLUME_PROFILE_BINS,
    lookback: int = VOLUME_PROFILE_LOOKBACK,
) -> None:
    """
    Compute Volume Profile: High Volume Nodes (HVN) and Low Volume Nodes (LVN).

    HVN = price levels with abnormally high historical volume.
    LVN = price levels with abnormally low historical volume.

    Adds columns:
      - VP_HVN_Min, VP_HVN_Max: the nearest HVN range
      - VP_LVN_Min, VP_LVN_Max: the nearest LVN range
      - DistToHVN_Pct: % distance from current close to HVN centre
      - DistToLVN_Pct: % distance from current close to LVN centre
      - Above_Nearest_HVN: True if close > HVN centre
    """
    if len(df) < lookback:
        lookback = len(df)

    subset = df.iloc[-lookback:]
    close, high, low, vol = subset["Close"], subset["High"], subset["Low"], subset["Volume"]

    # Build the volume profile
    price_min, price_max = low.min(), high.max()
    if price_min == price_max:
        # Degenerate case — single price
        df["VP_HVN_Center"] = price_min
        df["DistToHVN_Pct"] = 0.0
        return

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    profile = np.zeros(bins)

    # Accumulate volume across price bins
    for i in range(len(subset)):
        row_low, row_high = low.iloc[i], high.iloc[i]
        if row_low >= row_high:
            continue
        row_vol = vol.iloc[i]
        # Distribute volume across bins this bar spans
        bin_indices = np.digitize([row_low, row_high], bin_edges) - 1
        lo_idx = max(0, min(bin_indices[0], bins - 1))
        hi_idx = max(0, min(bin_indices[1], bins - 1))
        if hi_idx <= lo_idx:
            hi_idx = min(lo_idx + 1, bins - 1)
        for b in range(lo_idx, hi_idx + 1):
            profile[b] += row_vol / (hi_idx - lo_idx + 1)

    if profile.sum() == 0:
        return

    # Identify HVN (top tercile) and LVN (bottom tercile)
    threshold_hvn = np.percentile(profile[profile > 0], 67) if (profile > 0).any() else 0
    threshold_lvn = np.percentile(profile[profile > 0], 33) if (profile > 0).any() else 0

    hvn_mask = profile >= threshold_hvn
    lvn_mask = (profile > 0) & (profile <= threshold_lvn)

    current_price = close.iloc[-1]

    # Nearest HVN
    if hvn_mask.any():
        hvn_centers = bin_centers[hvn_mask]
        hvn_vols = profile[hvn_mask]
        # Weighted HVN centre (by volume)
        weighted_hvn = np.average(hvn_centers, weights=hvn_vols)
        df["VP_HVN_Center"] = weighted_hvn
        df["DistToHVN_Pct"] = ((current_price - weighted_hvn) / weighted_hvn) * 100
        df["Above_HVN"] = current_price > weighted_hvn
    else:
        df["VP_HVN_Center"] = np.nan
        df["DistToHVN_Pct"] = np.nan
        df["Above_HVN"] = np.nan

    # Nearest LVN
    if lvn_mask.any():
        lvn_centers = bin_centers[lvn_mask]
        lvn_vols = profile[lvn_mask]
        weighted_lvn = np.average(lvn_centers, weights=lvn_vols)
        df["VP_LVN_Center"] = weighted_lvn
        df["DistToLVN_Pct"] = ((current_price - weighted_lvn) / weighted_lvn) * 100
    else:
        df["VP_LVN_Center"] = np.nan
        df["DistToLVN_Pct"] = np.nan


# ======================================================================
# Wyckoff Phase Detection
# ======================================================================

_WYCKOFF_STAGES = Literal[
    "Accumulation", "Markup", "Distribution", "Markdown",
    "Selling Climax", "Automatic Rally", "Secondary Test",
    "Unknown",
]


def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    """
    Automatic Wyckoff Phase Detection.

    Simplified decision tree based on:
    - Trend (MA200 direction, price position)
    - Volume behaviour
    - Price action (ATR, recent swings)

    Possible outputs: Accumulation, Markup, Distribution, Markdown,
                      Selling Climax, Automatic Rally, Secondary Test, Unknown.

    The result is written to df['WyckoffPhase'].
    """
    close, vol = df["Close"], df["Volume"]

    # Guard: need at least 200 bars
    if len(df) < 200:
        df["WyckoffPhase"] = "Unknown"
        return

    # Current and recent values
    price_now = close.iloc[-1]
    ma200 = close.rolling(200, min_periods=100).mean().iloc[-1]
    ma50 = close.rolling(50, min_periods=25).mean().iloc[-1]
    ma20 = close.rolling(20, min_periods=10).mean().iloc[-1]

    # MA200 slope over 60 days
    ma200_series = close.rolling(200, min_periods=100).mean()
    ma200_slope = _rolling_slope(ma200_series, 60).iloc[-1]

    # Volume context
    vol_ma20 = vol.rolling(20, min_periods=10).mean().iloc[-1]
    vol_ma60 = vol.rolling(60, min_periods=30).mean().iloc[-1]
    vol_spike = vol_ma20 > vol_ma60 * 1.5

    # ATR context
    if "ATR14" not in df.columns:
        compute_atr(df)
    atr14 = df["ATR14"].iloc[-1]
    atr50 = df.get("ATR50", pd.Series(np.nan, index=df.index))
    atr50_now = atr50.iloc[-1] if len(atr50) > 0 else atr14
    atr_contracting = atr14 < atr50_now * 0.85

    # Price extremes over last 60 days
    high_60d = close.iloc[-60:].max()
    low_60d = close.iloc[-60:].min()
    dist_from_low = (price_now - low_60d) / low_60d * 100 if low_60d > 0 else 0
    dist_from_high = (high_60d - price_now) / high_60d * 100 if high_60d > 0 else 0

    # 52-week extremes
    high_52w = df.get("High52W", pd.Series(np.nan, index=df.index))
    low_52w = df.get("Low52W", pd.Series(np.nan, index=df.index))
    h52 = high_52w.iloc[-1] if len(high_52w) > 0 else high_60d
    l52 = low_52w.iloc[-1] if len(low_52w) > 0 else low_60d
    near_52w_low = (price_now - l52) / l52 * 100 < 5 if l52 > 0 else False
    near_52w_high = (h52 - price_now) / h52 * 100 < 5 if h52 > 0 else False

    # Volatility context
    hv20 = df.get("HV20", pd.Series(np.nan, index=df.index)).iloc[-1] if "HV20" in df.columns else np.nan
    hv60 = df.get("HV60", pd.Series(np.nan, index=df.index)).iloc[-1] if "HV60" in df.columns else np.nan
    vol_contracting = (not np.isnan(hv20) and not np.isnan(hv60) and hv20 < hv60 * 0.85)

    # ---- Decision Tree ----

    phase = "Unknown"

    # Selling Climax: price near 52w low, volume spike, ATR elevated, then contracting
    if near_52w_low and vol_spike:
        # Check if there was a sell-off recently
        recent_decline = (close.iloc[-20] - low_60d) / close.iloc[-20] * 100 > 15 if close.iloc[-20] > 0 else False
        if recent_decline:
            phase = "Selling Climax"

    # Automatic Rally: coming off a selling climax, price rising from lows
    if phase == "Unknown" and dist_from_low > 5 and dist_from_high > 15:
        recent_lowest = close.iloc[-60:].idxmin()
        if recent_lowest is not None:
            bars_from_low = len(df) - df.index.get_loc(recent_lowest) - 1
            if 5 <= bars_from_low <= 30:
                phase = "Automatic Rally"

    # Secondary Test: price retests the low on lower volume
    if phase == "Unknown" and dist_from_low < 10 and not vol_spike and atr_contracting:
        # Check if there was a prior rally
        recent_high = close.iloc[-60:-10].max() if len(df) >= 70 else close.iloc[-60:].max()
        rallied = (recent_high - low_60d) / low_60d * 100 > 10 if low_60d > 0 else False
        if rallied:
            phase = "Secondary Test"

    # Accumulation: price above lows, below MA200, volume increasing, low volatility
    if phase == "Unknown":
        if (
            price_now < ma200
            and ma200_slope <= 0
            and vol_ma20 > vol_ma60
            and atr_contracting
            and vol_contracting
            and dist_from_low < 20
            and not near_52w_high
        ):
            phase = "Accumulation"

    # Markup: price above MA200, MA200 rising, higher prices
    if phase == "Unknown":
        if price_now > ma200 and ma200_slope > 0 and dist_from_high < 20:
            phase = "Markup"

    # Distribution: near 52w high, volume elevated, ATR expanding
    if phase == "Unknown":
        if (
            near_52w_high
            and vol_ma20 > vol_ma60
            and not atr_contracting
            and ma200_slope < 0.5  # losing momentum
        ):
            phase = "Distribution"

    # Markdown: below MA200, MA200 declining, volume elevated on down days
    if phase == "Unknown":
        if price_now < ma200 and ma200_slope < 0 and dist_from_low > 30 and not atr_contracting:
            phase = "Markdown"

    df["WyckoffPhase"] = phase


# ======================================================================
# Master function — compute everything for one ticker
# ======================================================================

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator on *df* (mutates in-place and returns it).

    This is the single entry point for the scanner — call once per ticker.
    """
    if df.empty or len(df) < 20:
        return df

    # Price
    compute_moving_averages(df)
    compute_ema(df)
    compute_atr(df)
    compute_adx(df)
    compute_cci(df)
    compute_roc(df)
    compute_52week_levels(df)

    # Volume
    compute_volume_mas(df)
    compute_volume_ratios(df)
    compute_relative_volume(df)
    compute_volume_zscore(df)
    compute_volume_trend(df)

    # Money Flow
    compute_obv(df)
    compute_obv_slope(df)
    compute_ad_line(df)
    compute_ad_slope(df)
    compute_cmf(df)
    compute_mfi(df)
    compute_vwap(df)

    # Trend
    compute_regression(df)
    compute_macd(df)
    compute_rsi(df)

    # Volatility
    compute_historical_volatility(df)
    compute_atr_compression(df)
    compute_bollinger_bands(df)
    compute_donchian(df)

    # Volume Profile
    try:
        compute_volume_profile(df)
    except Exception:
        logger.debug("Volume Profile failed for a ticker — skipping.", exc_info=True)

    # Wyckoff
    try:
        detect_wyckoff_phase(df)
    except Exception:
        logger.debug("Wyckoff detection failed for a ticker — skipping.", exc_info=True)

    return df
