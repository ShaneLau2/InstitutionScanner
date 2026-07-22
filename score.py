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

logger = logging.getLogger("scanner_gui.score")


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
    if len(df) < 252 or "Close" not in df.columns or "MA200" not in df.columns:
        return 0.0

    close = df["Close"].replace([np.inf, -np.inf], np.nan).dropna()
    ma200 = df["MA200"].reindex(close.index).replace([np.inf, -np.inf], np.nan)
    valid = pd.concat({"close": close, "ma200": ma200}, axis=1).dropna()
    if len(valid) < 60:
        return 0.0

    price_now = float(valid["close"].iloc[-1])
    ma200_now = float(valid["ma200"].iloc[-1])
    if price_now <= 0 or ma200_now <= 0:
        return 0.0

    score = 0.0
    ma_recent = valid["ma200"].iloc[-60:]
    slope_pct = (float(ma_recent.iloc[-1]) / float(ma_recent.iloc[0])) - 1.0
    if slope_pct < 0:
        score += _clamp(abs(slope_pct) / 0.12) * 5.0

    below_pct = (ma200_now - price_now) / ma200_now
    if below_pct > 0:
        score += _clamp(below_pct / 0.30) * 6.0
        score -= _clamp(max(below_pct - 0.45, 0.0) / 0.30) * 3.0

    below = valid["close"] < valid["ma200"]
    last_above = np.flatnonzero((~below).to_numpy())
    days_below = len(valid) - int(last_above[-1]) - 1 if len(last_above) else len(valid)
    score += _clamp(days_below / 250.0) * 3.0

    lookback = valid["close"].iloc[-min(504, len(valid)):]
    peak = float(lookback.max())
    drawdown = (price_now - peak) / peak if peak > 0 else 0.0
    depth = abs(drawdown)
    if 0.15 <= depth <= 0.50:
        score += _clamp(1.0 - abs(depth - 0.32) / 0.25) * 3.0

    recovery = valid["close"].iloc[-20:]
    if len(recovery) >= 10:
        recent_slope = float(recovery.iloc[-1] / recovery.iloc[0] - 1.0)
        if recent_slope > 0:
            score += _clamp(recent_slope / 0.12) * 3.0

    return _clamp(score, 0.0, 20.0)


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

    if "VolMA20" in df.columns and "VolMA120" in df.columns:
        vol_ma20 = df["VolMA20"].replace([np.inf, -np.inf], np.nan)
        vol_ma120 = df["VolMA120"].replace([np.inf, -np.inf], np.nan)
        ratio_series = (vol_ma20 / vol_ma120.replace(0, np.nan)).dropna()
        if len(ratio_series) >= VOLUME_ACCUM_MIN_DAYS:
            consecutive = 0
            for value in ratio_series.iloc[::-1]:
                if value >= VOLUME_ACCUM_RATIO:
                    consecutive += 1
                else:
                    break
            if consecutive >= VOLUME_ACCUM_MIN_DAYS:
                score += 4.0 + _clamp((consecutive - VOLUME_ACCUM_MIN_DAYS) / 80.0) * 6.0
            ratio_now = float(ratio_series.iloc[-1])
            score += _clamp((ratio_now - VOLUME_ACCUM_RATIO) / 0.8) * 3.0

            if len(ratio_series) >= 20:
                ratio_change = float(ratio_series.iloc[-1] - ratio_series.iloc[-20])
                score += _clamp(ratio_change / 0.5) * 4.0

    if "VolZScore" in df.columns:
        z_recent = df["VolZScore"].replace([np.inf, -np.inf], np.nan).dropna().iloc[-30:]
        if len(z_recent) >= 10:
            z_now = float(z_recent.iloc[-1])
            positive_days = float((z_recent > 0).mean())
            score += positive_days * 3.0
            score += _clamp(z_now / 2.0) * 2.0

    return _clamp(score, 0.0, 25.0)


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

    if "OBV" in df.columns and len(df) >= 60:
        recent = df[["Close", "OBV"]].iloc[-60:].replace([np.inf, -np.inf], np.nan).dropna()
        if len(recent) >= 40:
            split = len(recent) // 2
            first_half = recent.iloc[:split]
            second_half = recent.iloc[split:]
            first_price_low = float(first_half["Close"].min())
            second_price_low = float(second_half["Close"].min())
            first_obv_low = float(first_half["OBV"].min())
            second_obv_low = float(second_half["OBV"].min())
            price_now = float(recent["Close"].iloc[-1])
            obv_now = float(recent["OBV"].iloc[-1])
            near_low = second_price_low > 0 and (price_now - second_price_low) / second_price_low <= 0.05
            price_retest = second_price_low <= first_price_low * 1.02
            obv_divergence = second_obv_low > first_obv_low and obv_now >= second_obv_low
            if near_low and price_retest and obv_divergence:
                score += 8.0
            elif obv_divergence:
                score += 3.0

    if "AD" in df.columns and "AD_Slope" in df.columns:
        ad = df["AD"].replace([np.inf, -np.inf], np.nan).dropna()
        ad_slope = df["AD_Slope"].iloc[-1]
        if len(ad) >= AD_SLOPE_LOOKBACK and pd.notna(ad_slope):
            ad_scale = max(float(ad.iloc[-AD_SLOPE_LOOKBACK:].abs().median()), 1.0)
            slope_score = _clamp(float(ad_slope) / (ad_scale * 0.03))
            score += slope_score * 5.0
            if float(ad.iloc[-1]) >= float(ad.iloc[-min(120, len(ad)):].max()) * 0.95:
                score += 1.0

    if "CMF" in df.columns:
        cmf = df["CMF"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(cmf) >= 20:
            cmf_now = float(cmf.iloc[-1])
            cmf_change = cmf_now - float(cmf.iloc[-20])
            score += _clamp(cmf_now / 0.15) * 4.0
            score += _clamp(cmf_change / 0.10) * 2.0

    if "MFI" in df.columns:
        mfi_now = df["MFI"].replace([np.inf, -np.inf], np.nan).iloc[-1]
        if pd.notna(mfi_now):
            score += 3.0 if 40 <= mfi_now <= 70 else 1.5 if 30 <= mfi_now <= 80 else 0.0

    return _clamp(score, 0.0, 25.0)


# ======================================================================
# Volatility Score (15 points)
# ======================================================================

def score_volatility(df: pd.DataFrame) -> float:
    if len(df) < BB_WIDTH_COMPRESSION_LOOKBACK:
        return 0.0

    components: list[float] = []
    if "ATR14" in df.columns and "ATR50" in df.columns:
        atr14 = df["ATR14"].replace([np.inf, -np.inf], np.nan).iloc[-1]
        atr50 = df["ATR50"].replace([np.inf, -np.inf], np.nan).iloc[-1]
        if pd.notna(atr14) and pd.notna(atr50) and atr50 > 0:
            components.append(_clamp((1.0 - float(atr14 / atr50)) / 0.35))

    if "BB_Width" in df.columns:
        bb = df["BB_Width"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(bb) >= BB_WIDTH_COMPRESSION_LOOKBACK:
            current = float(bb.iloc[-1])
            baseline = float(bb.iloc[-BB_WIDTH_COMPRESSION_LOOKBACK:-10].median())
            if baseline > 0:
                components.append(_clamp(1.0 - current / baseline))

    if "HV20" in df.columns and "HV60" in df.columns:
        hv20 = df["HV20"].replace([np.inf, -np.inf], np.nan).iloc[-1]
        hv60 = df["HV60"].replace([np.inf, -np.inf], np.nan).iloc[-1]
        if pd.notna(hv20) and pd.notna(hv60) and hv60 > 0:
            components.append(_clamp((1.0 - float(hv20 / hv60)) / 0.5))

    if not components:
        return 0.0
    coverage = len(components) / 3.0
    return _clamp(float(np.mean(components)) * coverage) * 15.0


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
            if 0 <= dist_low <= 20:
                if dist_low < 8:
                    score += dist_low / 8 * 5
                elif dist_low <= 12:
                    score += 5
                else:
                    score += (20 - dist_low) / 8 * 5

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

def classify_style(df: pd.DataFrame, is_etf: bool = False) -> str:
    if is_etf:
        return "ETF趋势/资金"
    if len(df) < 60:
        return "数据不足"
    close = df["Close"]
    atr = float(df["ATR14"].iloc[-1]) if "ATR14" in df.columns else np.nan
    atr_pct = atr / float(close.iloc[-1]) if close.iloc[-1] > 0 and not np.isnan(atr) else np.nan
    roc = float(df["ROC21"].iloc[-1]) if "ROC21" in df.columns and not np.isnan(df["ROC21"].iloc[-1]) else 0.0
    volume_ratio = 1.0
    if "VolMA20" in df.columns and "VolMA120" in df.columns and df["VolMA120"].iloc[-1] > 0:
        volume_ratio = float(df["VolMA20"].iloc[-1] / df["VolMA120"].iloc[-1])
    if not np.isnan(atr_pct) and atr_pct >= 0.045:
        return "高波动成长"
    if roc >= 12:
        return "趋势成长"
    if volume_ratio >= 1.25:
        return "资金吸筹"
    if not np.isnan(atr_pct) and atr_pct <= 0.025:
        return "低波动防守"
    return "均衡"


def _style_adjustment(df: pd.DataFrame, style: str) -> tuple[float, float, float, float, float]:
    if style == "高波动成长":
        return (1.15, 1.05, 0.90, 0.85, 0.95)
    if style == "趋势成长":
        return (1.25, 1.00, 0.90, 0.85, 0.95)
    if style == "资金吸筹":
        return (0.90, 1.05, 1.25, 1.05, 1.00)
    if style == "低波动防守":
        return (0.90, 0.95, 1.05, 1.25, 1.20)
    if style == "ETF趋势/资金":
        return (1.00, 1.00, 1.10, 1.00, 0.90)
    return (1.00, 1.00, 1.00, 1.00, 1.00)


def score_ticker(df: pd.DataFrame, is_etf: bool = False) -> ScoreBreakdown:
    """
    Compute the full accumulation score for one ticker.

    Args:
        df: DataFrame with all indicators pre-computed.

    Returns:
        ScoreBreakdown with total and sub-scores.
    """
    raw_scores = (
        score_trend(df),
        score_volume(df),
        score_accumulation(df),
        score_volatility(df),
        score_structure(df),
    )
    style = classify_style(df, is_etf=is_etf)
    adjustments = _style_adjustment(df, style)
    limits = (20.0, 25.0, 25.0, 15.0, 15.0)
    adjusted_scores = tuple(
        _clamp(score * adjustment, 0.0, limit)
        for score, adjustment, limit in zip(raw_scores, adjustments, limits)
    )
    trend, volume, accumulation, volatility, structure = adjusted_scores
    total = sum(adjusted_scores)

    return ScoreBreakdown(
        total=total,
        trend=trend,
        volume=volume,
        accumulation=accumulation,
        volatility=volatility,
        structure=structure,
    )
