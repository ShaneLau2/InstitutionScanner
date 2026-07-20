"""
score.py — Institutional Accumulation Scoring Engine.

Scores each ticker on a 0–100 scale across five dimensions:
  - Trend          (20 points):  Bear market characteristics
  - Volume         (25 points):  Sustained above-average volume
  - Accumulation   (25 points):  OBV, A/D, CMF, MFI signals
  - Volatility     (15 points):  ATR & BB compression
  - Structure      (15 points):  Distance from lows, consolidation duration

The scoring is designed so that higher scores mean stronger
institutional accumulation signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AD_SLOPE_LOOKBACK,
    ATR_COMPRESSION_LOOKBACK,
    BB_WIDTH_COMPRESSION_LOOKBACK,
    CONSOLIDATION_DAYS,
    VOLUME_ACCUM_MIN_DAYS,
    VOLUME_ACCUM_RATIO,
    VOLUME_PROFILE_LOOKBACK,
)

logger = logging.getLogger("institution_scanner.score")


# ======================================================================
# Score Result
# ======================================================================

@dataclass
class ScoreBreakdown:
    """Full scoring output for one ticker."""
    total: float = 0.0
    trend: float = 0.0
    volume: float = 0.0
    accumulation: float = 0.0
    volatility: float = 0.0
    structure: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "Score": round(self.total, 2),
            "TrendScore": round(self.trend, 2),
            "VolumeScore": round(self.volume, 2),
            "AccumulationScore": round(self.accumulation, 2),
            "CompressionScore": round(self.volatility, 2),
            "StructureScore": round(self.structure, 2),
        }


# ======================================================================
# Sub-score helpers
# ======================================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def _normalize_to_range(
    value: float, min_val: float, max_val: float, invert: bool = False
) -> float:
    """Normalize *value* into [0, 1] given [min_val, max_val].  If *invert*, 1 is worst."""
    if max_val == min_val:
        return 0.5
    norm = (value - min_val) / (max_val - min_val)
    norm = _clamp(norm)
    return 1.0 - norm if invert else norm


# ======================================================================
# Trend Score (20 points)
# ======================================================================

def score_trend(df: pd.DataFrame) -> float:
    """
    Score the bear market trend.

    Higher score = deeper, more established bear trend
    (stronger contrarian setup for accumulation).

    Max: 20 points
    """

    if len(df) < 252:
        return 0.0

    close = df["Close"]
    price_now = close.iloc[-1]
    ma200 = df.get("MA200", pd.Series(np.nan, index=df.index))
    ma200_now = ma200.iloc[-1] if not ma200.empty else price_now

    score = 0.0

    # 1. MA200 declining? (up to 7 points)
    if "MA200" in df.columns:
        ma200_recent = df["MA200"].dropna().iloc[-60:]
        if len(ma200_recent) >= 30 and ma200_recent.iloc[0] > 0:
            decline_pct = (ma200_recent.iloc[-1] - ma200_recent.iloc[0]) / ma200_recent.iloc[0]
            if decline_pct < 0:
                # Score based on severity of decline: 0% to -15% maps to 0-7
                score += _clamp(abs(decline_pct) / 0.15, 0, 1) * 7

    # 2. Price below MA200? (up to 6 points)
    if ma200_now > 0:
        pct_below = (ma200_now - price_now) / ma200_now  # positive = below
        if pct_below > 0:
            score += _clamp(pct_below / 0.20, 0, 1) * 6  # 20% below MA200 = max

    # 3. Bear market duration (up to 4 points)
    # Days since price was last above MA200
    if "MA200" in df.columns:
        above_ma200 = close > df["MA200"]
        if above_ma200.any():
            last_above_idx = above_ma200[above_ma200].index[-1]
            days_below = len(df) - df.index.get_loc(last_above_idx) - 1
            # 0 → 500+ days maps to 0–4 points
            score += _clamp(days_below / 500, 0, 1) * 4
        else:
            score += 4  # always below MA200

    # 4. Severity of decline from 2-year high (up to 3 points)
    lookback = min(504, len(df))  # ~2 years
    peak = close.iloc[-lookback:].max()
    if peak > 0:
        drawdown = (price_now - peak) / peak
        if drawdown < 0:
            # -30% to -80% maps to 0–3
            score += _clamp(abs(drawdown - 0.3) / 0.5, 0, 1) * 3

    return min(score, 20.0)


# ======================================================================
# Volume Score (25 points)
# ======================================================================

def score_volume(df: pd.DataFrame) -> float:
    """
    Score the volume accumulation signal.

    Higher score = stronger, more sustained volume expansion.

    Max: 25 points
    """

    if len(df) < 120:
        return 0.0

    score = 0.0

    # 1. Volume Ratio persistence (up to 10 points)
    if "VolMA20" in df.columns and "VolMA120" in df.columns:
        vol_ma20 = df["VolMA20"]
        vol_ma120 = df["VolMA120"]
        condition = vol_ma20 > vol_ma120 * VOLUME_ACCUM_RATIO

        # Count consecutive days
        consecutive = 0
        for i in range(len(condition) - 1, -1, -1):
            if condition.iloc[i]:
                consecutive += 1
            else:
                break

        # VOLUME_ACCUM_MIN_DAYS to 120+ days maps to 0–10
        normalised = _normalize_to_range(consecutive, VOLUME_ACCUM_MIN_DAYS, 120)
        score += normalised * 10

        # Bonus: current ratio magnitude
        if vol_ma120.iloc[-1] > 0:
            ratio = vol_ma20.iloc[-1] / vol_ma120.iloc[-1]
            # 1.0 → 2.5+ ratio maps to 0–3 bonus
            bonus = _clamp((ratio - 1.0) / 1.5, 0, 1) * 3
            score += bonus

    # 2. Volume Trend (up to 7 points)
    if "VolTrend" in df.columns:
        vol_trend = df["VolTrend"].iloc[-1]
        if not np.isnan(vol_trend):
            # Positive and strong trend = good
            # Normalise: 0 to large positive
            score += _clamp(max(vol_trend, 0) / 10000, 0, 1) * 7

    # 3. Volume stability (up to 5 points)
    # Lower Z-score variance = more consistent buying
    if "VolZScore" in df.columns:
        z_recent = df["VolZScore"].dropna().iloc[-30:]
        if len(z_recent) > 10:
            z_volatility = z_recent.std()
            # Low volatility (stable buying) scores high
            # Std 0 → 2 maps to 5 → 0
            score += _clamp(1 - z_volatility / 2, 0, 1) * 5

    # 4. Volume Z-score currently positive (up to 3 points)
    if "VolZScore" in df.columns:
        z_now = df["VolZScore"].iloc[-1]
        if not np.isnan(z_now) and z_now > 0:
            score += _clamp(z_now / 2, 0, 1) * 3

    return min(score, 25.0)


# ======================================================================
# Accumulation Score (25 points)
# ======================================================================

def score_accumulation(df: pd.DataFrame) -> float:
    """
    Score accumulation signals from OBV, A/D, CMF, MFI.

    Higher score = stronger evidence of institutional buying.

    Max: 25 points
    """

    if len(df) < 60:
        return 0.0

    score = 0.0

    # 1. OBV Divergence (up to 8 points)
    if "OBV" in df.columns and len(df) >= 60:
        close, obv = df["Close"], df["OBV"]
        # Find lowest close and corresponding OBV
        price_low = close.iloc[-60:].min()
        price_low_idx = close.iloc[-60:].idxmin()
        obv_at_low = obv.loc[price_low_idx]
        obv_now = obv.iloc[-1]

        if price_low > 0 and obv_at_low != 0:
            price_near_low = (close.iloc[-1] - price_low) / price_low < 0.05
            obv_divergence = (obv_now - obv_at_low) / abs(obv_at_low) if obv_at_low != 0 else 0

            if price_near_low and obv_divergence > 0.02:
                # Strong divergence: price near low, OBV much higher
                score += _clamp(obv_divergence / 0.10, 0, 1) * 8
            elif obv_divergence > 0.02:
                score += _clamp(obv_divergence / 0.10, 0, 1) * 4

    # 2. A/D Line strength (up to 8 points)
    if "AD_Slope" in df.columns:
        ad_slope = df["AD_Slope"].iloc[-1]
        if not np.isnan(ad_slope):
            # Positive slope = accumulation
            if ad_slope > 0:
                score += _clamp(ad_slope / 5e6, 0, 1) * 6
            # A/D line is making new highs while price is not
            if "AD" in df.columns:
                ad_now = df["AD"].iloc[-1]
                ad_max = df["AD"].max()
                if ad_max > 0 and ad_now >= ad_max * 0.95:
                    # Near all-time high A/D while price is weak = strong accumulation
                    score += 2

    # 3. CMF (up to 6 points)
    if "CMF" in df.columns:
        cmf_now = df["CMF"].iloc[-1]
        if not np.isnan(cmf_now):
            # CMF > 0 = accumulation
            score += _clamp(cmf_now / 0.15, 0, 1) * 4
            # CMF improving trend
            cmf_20d_ago = df["CMF"].iloc[-20] if len(df) >= 20 else cmf_now
            cmf_improvement = cmf_now - cmf_20d_ago
            score += _clamp(cmf_improvement / 0.10, 0, 1) * 2

    # 4. MFI (up to 3 points)
    if "MFI" in df.columns:
        mfi_now = df["MFI"].iloc[-1]
        if not np.isnan(mfi_now):
            # MFI in 40-70 range during bear market = quiet accumulation
            if 40 <= mfi_now <= 70:
                score += 3
            elif 30 <= mfi_now <= 80:
                score += 1.5

    return min(score, 25.0)


# ======================================================================
# Volatility Score (15 points)
# ======================================================================

def score_volatility(df: pd.DataFrame) -> float:
    """
    Score volatility contraction ("coiled spring").

    Higher score = stronger compression = more likely breakout.

    Max: 15 points
    """

    if len(df) < BB_WIDTH_COMPRESSION_LOOKBACK:
        return 0.0

    score = 0.0

    # 1. ATR Compression (up to 7 points)
    if "ATR14" in df.columns and "ATR50" in df.columns:
        atr14 = df["ATR14"].iloc[-1]
        atr50 = df["ATR50"].iloc[-1]
        if atr50 > 0:
            atr_ratio = atr14 / atr50
            # Ratio < 0.7 = strong compression
            if atr_ratio < 1.0:
                score += _clamp((1.0 - atr_ratio) / 0.4, 0, 1) * 7

    # 2. BB Width compression (up to 5 points)
    if "BB_Width" in df.columns:
        bb_width = df["BB_Width"].dropna()
        if len(bb_width) >= BB_WIDTH_COMPRESSION_LOOKBACK:
            recent = bb_width.iloc[-BB_WIDTH_COMPRESSION_LOOKBACK:]
            if recent.iloc[0] > 0:
                decline = (recent.iloc[0] - recent.iloc[-1]) / recent.iloc[0]
                if decline > 0:
                    score += _clamp(decline / 0.5, 0, 1) * 5

    # 3. HV contraction (up to 3 points)
    if "HV20" in df.columns and "HV60" in df.columns:
        hv20 = df["HV20"].iloc[-1]
        hv60 = df["HV60"].iloc[-1]
        if not np.isnan(hv20) and not np.isnan(hv60) and hv60 > 0:
            hv_ratio = hv20 / hv60
            if hv_ratio < 1.0:
                score += _clamp((1.0 - hv_ratio) / 0.5, 0, 1) * 3

    return min(score, 15.0)


# ======================================================================
# Structure Score (15 points)
# ======================================================================

def score_structure(df: pd.DataFrame) -> float:
    """
    Score the structural bottom formation.

    Higher score = more established bottom structure.

    Max: 15 points
    """

    if len(df) < 252:
        return 0.0

    close = df["Close"]
    price_now = close.iloc[-1]
    score = 0.0

    # 1. Distance from 52-week low (up to 5 points)
    if "Low52W" in df.columns and "DistToLow52W" in df.columns:
        dist_low = df["DistToLow52W"].iloc[-1]
        if not np.isnan(dist_low):
            # Close to 52w low but not at it = ideal
            # 0% → 20% maps to 5 → 0 (closer = better)
            if 0 <= dist_low <= 20:
                score += _clamp(1 - dist_low / 20, 0, 1) * 5

    # 2. Consolidation duration (up to 5 points)
    # How long has price been range-bound near the bottom?
    if len(df) >= CONSOLIDATION_DAYS:
        recent = df.iloc[-CONSOLIDATION_DAYS:]
        high, low = recent["High"].max(), recent["Low"].min()
        avg_price = recent["Close"].mean()
        if avg_price > 0:
            range_pct = (high - low) / avg_price * 100
            if range_pct <= 15:
                # Tighter + longer = up to 5 points
                # Use the range tightness
                score += _clamp(1 - range_pct / 15, 0, 1) * 5

    # 3. Linear regression slope near zero (up to 3 points)
    if "RegSlope" in df.columns:
        reg_slope = df["RegSlope"].iloc[-1]
        if not np.isnan(reg_slope):
            # Slope near 0 + high R² = stable base
            abs_slope = abs(reg_slope)
            # 0 → very flat base
            score += _clamp(1 - abs_slope / 0.05, 0, 1) * 2

            # R² bonus
            if "RegR2" in df.columns:
                r2 = df["RegR2"].iloc[-1]
                if not np.isnan(r2):
                    score += _clamp(r2, 0, 1) * 1

    # 4. Volume Profile — price above HVN (up to 2 points)
    # Price sitting above a High Volume Node = potential support
    if "Above_HVN" in df.columns and "DistToHVN_Pct" in df.columns:
        above_hvn = df["Above_HVN"].iloc[-1]
        dist_hvn = df["DistToHVN_Pct"].iloc[-1]
        if above_hvn is True and not np.isnan(dist_hvn):
            # Small positive distance (just above HVN) = ideal
            if 0 < dist_hvn < 10:
                score += _clamp(1 - dist_hvn / 10, 0, 1) * 2

    return min(score, 15.0)


# ======================================================================
# Master scoring function
# ======================================================================

def score_ticker(df: pd.DataFrame) -> ScoreBreakdown:
    """
    Compute the full accumulation score for one ticker.

    Args:
        df: DataFrame with all indicators pre-computed.

    Returns:
        ScoreBreakdown with total and sub-scores.
    """
    trend = score_trend(df)
    volume = score_volume(df)
    accumulation = score_accumulation(df)
    volatility = score_volatility(df)
    structure = score_structure(df)

    total = trend + volume + accumulation + volatility + structure

    return ScoreBreakdown(
        total=min(total, 100.0),
        trend=trend,
        volume=volume,
        accumulation=accumulation,
        volatility=volatility,
        structure=structure,
    )
