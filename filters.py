"""
filters.py — Screening filters for institutional accumulation detection.

Each filter is a standalone function that accepts a DataFrame (with all
indicators pre-computed) and returns a bool or a structured result dict.

Filters are lightweight — they operate on already-computed indicator
columns, not raw prices. This keeps the scanner fast.
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
    BEAR_DECLINE_PCT,
    BEAR_LOOKBACK_YEARS,
    BEAR_MA200_DECLINING_DAYS,
    CMF_THRESHOLD,
    CONSOLIDATION_DAYS,
    CONSOLIDATION_MAX_RANGE_PCT,
    MIN_MARKET_CAP,
    MIN_PRICE,
    MIN_VOLUME,
    OBV_DIVERGENCE_LOOKBACK,
    VOLUME_ACCUM_MIN_DAYS,
    VOLUME_ACCUM_RATIO,
)

logger = logging.getLogger("institution_scanner.filters")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Container returned by each filter function."""
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Basic sanity filters
# ======================================================================

def filter_min_price(df: pd.DataFrame) -> FilterResult:
    """Reject if latest close is below MIN_PRICE."""
    close = df["Close"].iloc[-1]
    passed = close >= MIN_PRICE
    return FilterResult(
        passed=passed,
        reason=f"Close ${close:.2f} {'>=' if passed else '<'} MIN_PRICE ${MIN_PRICE:.2f}",
        details={"close": close},
    )


def filter_min_volume(df: pd.DataFrame) -> FilterResult:
    """Reject if average daily volume (60d) is below MIN_VOLUME."""
    vol_avg = df["Volume"].rolling(60, min_periods=30).mean().iloc[-1]
    passed = vol_avg >= MIN_VOLUME
    return FilterResult(
        passed=passed,
        reason=f"AvgVol {vol_avg:,.0f} {'>=' if passed else '<'} MIN_VOLUME {MIN_VOLUME:,}",
        details={"avg_volume_60": vol_avg},
    )


def filter_sufficient_history(df: pd.DataFrame) -> FilterResult:
    """Require at least 252 trading days (1 year) of data."""
    passed = len(df) >= 252
    return FilterResult(
        passed=passed,
        reason=f"{len(df)} bars {'>=' if passed else '<'} 252 minimum",
        details={"bars": len(df)},
    )


# ======================================================================
# Bear Market Detection
# ======================================================================

def filter_bear_market(df: pd.DataFrame) -> FilterResult:
    """
    Detect long-term bear market:
    1. MA200 must be declining (last 60 days slope < 0).
    2. Price must be below MA200.
    3. Price must have declined > 30% over past 2 years.

    Returns:
        FilterResult with details about the bear market.
    """
    details: dict[str, Any] = {}

    close = df["Close"]
    if len(df) < 252:
        return FilterResult(passed=False, reason="Insufficient history for bear market check")

    # 1. MA200 declining
    if "MA200" not in df.columns:
        return FilterResult(passed=False, reason="MA200 not computed")

    ma200 = df["MA200"]
    ma200_now = ma200.iloc[-1]
    price_now = close.iloc[-1]

    # Check MA200 slope over recent days
    ma200_recent = ma200.dropna().iloc[-BEAR_MA200_DECLINING_DAYS:]
    if len(ma200_recent) < 10:
        return FilterResult(passed=False, reason="Not enough MA200 data")

    # Simple slope: last value vs first value
    ma200_declining = ma200_recent.iloc[-1] < ma200_recent.iloc[0]
    details["ma200_now"] = round(ma200_now, 2)
    details["ma200_declining"] = ma200_declining
    details["ma200_slope_pct"] = round(
        (ma200_recent.iloc[-1] - ma200_recent.iloc[0]) / ma200_recent.iloc[0] * 100, 2
    )

    # 2. Price below MA200
    price_below_ma200 = price_now < ma200_now
    details["price_now"] = round(price_now, 2)
    details["price_below_ma200"] = price_below_ma200
    details["pct_below_ma200"] = round((price_now - ma200_now) / ma200_now * 100, 2)

    # 3. Decline over 2 years
    lookback_bars = min(BEAR_LOOKBACK_YEARS * 252, len(df))
    price_lookback = close.iloc[-lookback_bars]
    decline_pct = (price_now - price_lookback) / price_lookback * 100 if price_lookback > 0 else 0
    bear_decline = decline_pct <= BEAR_DECLINE_PCT
    details["decline_pct"] = round(decline_pct, 2)
    details["price_lookback"] = round(price_lookback, 2)

    passed = ma200_declining and price_below_ma200 and bear_decline

    reasons = []
    if not ma200_declining:
        reasons.append("MA200 not declining")
    if not price_below_ma200:
        reasons.append("Price above MA200")
    if not bear_decline:
        reasons.append(f"Decline {decline_pct:.1f}% > {BEAR_DECLINE_PCT}% threshold")

    reason = "Bear market confirmed" if passed else "; ".join(reasons)
    return FilterResult(passed=passed, reason=reason, details=details)


# ======================================================================
# Bottom Consolidation (Range-bound, Low Volatility)
# ======================================================================

def filter_consolidation(df: pd.DataFrame) -> FilterResult:
    """
    Detect bottom consolidation:
    - Over the last CONSOLIDATION_DAYS, the max price range (high-low) is
      less than CONSOLIDATION_MAX_RANGE_PCT percent.
    """
    if len(df) < CONSOLIDATION_DAYS:
        return FilterResult(passed=False, reason="Insufficient history for consolidation check")

    recent = df.iloc[-CONSOLIDATION_DAYS:]
    high = recent["High"].max()
    low = recent["Low"].min()
    avg_price = recent["Close"].mean()

    if avg_price <= 0:
        return FilterResult(passed=False, reason="Invalid average price")

    range_pct = (high - low) / avg_price * 100
    passed = range_pct <= CONSOLIDATION_MAX_RANGE_PCT

    return FilterResult(
        passed=passed,
        reason=(
            f"Range {range_pct:.1f}% {'<=' if passed else '>'} "
            f"{CONSOLIDATION_MAX_RANGE_PCT}% threshold"
        ),
        details={
            "range_pct": round(range_pct, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "consolidation_days": CONSOLIDATION_DAYS,
        },
    )


# ======================================================================
# Volume Accumulation Detection
# ======================================================================

def filter_volume_accumulation(df: pd.DataFrame) -> FilterResult:
    """
    Detect sustained volume accumulation:
    - VolMA20 > VolMA120 * VOLUME_ACCUM_RATIO for at least
      VOLUME_ACCUM_MIN_DAYS consecutive days.
    - Not a one-day spike — must be persistent.
    """
    if "VolMA20" not in df.columns or "VolMA120" not in df.columns:
        return FilterResult(passed=False, reason="Volume MAs not computed")

    if len(df) < VOLUME_ACCUM_MIN_DAYS + 120:
        return FilterResult(passed=False, reason="Insufficient history for volume check")

    vol_ma20 = df["VolMA20"]
    vol_ma120 = df["VolMA120"]

    # Boolean series: True where volume accumulation condition holds
    condition = vol_ma20 > vol_ma120 * VOLUME_ACCUM_RATIO
    condition = condition.fillna(False)

    # Count consecutive True days
    if not condition.iloc[-1]:
        return FilterResult(
            passed=False,
            reason=f"Volume accumulation not active today (VolMA20={vol_ma20.iloc[-1]:,.0f}, "
                   f"VolMA120={vol_ma120.iloc[-1]:,.0f})",
            details={
                "vol_ma20": vol_ma20.iloc[-1],
                "vol_ma120": vol_ma120.iloc[-1],
                "ratio": round(vol_ma20.iloc[-1] / vol_ma120.iloc[-1], 2) if vol_ma120.iloc[-1] > 0 else 0,
            },
        )

    # Walk backwards to count consecutive True
    consecutive = 0
    for i in range(len(condition) - 1, -1, -1):
        if condition.iloc[i]:
            consecutive += 1
        else:
            break

    passed = consecutive >= VOLUME_ACCUM_MIN_DAYS

    volumes = condition.reset_index(drop=True)
    consecutive_series = pd.Series(0, index=condition.index, dtype=int)
    cnt = 0
    for i in range(len(condition)):
        if condition.iloc[i]:
            cnt += 1
        else:
            cnt = 0
        consecutive_series.iloc[i] = cnt
    max_consecutive = int(consecutive_series.max())
    df["_VolAccumDays"] = consecutive_series  # temporary column for scoring

    return FilterResult(
        passed=passed,
        reason=(
            f"Volume accumulation: {consecutive} consecutive days "
            f"(need {VOLUME_ACCUM_MIN_DAYS}), "
            f"peak {max_consecutive}"
        ),
        details={
            "consecutive_days": consecutive,
            "max_consecutive": max_consecutive,
            "current_ratio": round(vol_ma20.iloc[-1] / vol_ma120.iloc[-1], 2) if vol_ma120.iloc[-1] > 0 else 0,
        },
    )


# ======================================================================
# OBV Bullish Divergence
# ======================================================================

def filter_obv_divergence(df: pd.DataFrame) -> FilterResult:
    """
    Detect OBV Bullish Divergence:
    - Price makes a new low (within lookback window).
    - OBV does NOT make a new low.
    - This suggests accumulation even as price is weak.
    """
    if "OBV" not in df.columns:
        return FilterResult(passed=False, reason="OBV not computed")

    if len(df) < OBV_DIVERGENCE_LOOKBACK:
        return FilterResult(passed=False, reason="Insufficient history for OBV divergence")

    lookback = min(OBV_DIVERGENCE_LOOKBACK, len(df))
    recent = df.iloc[-lookback:]

    close = recent["Close"]
    obv = recent["OBV"]

    # Find the lowest close in the window
    price_low_idx = close.idxmin()
    price_low = close.min()

    # OBV at the time of the price low
    obv_at_price_low = obv.loc[price_low_idx]

    # Current OBV
    obv_now = obv.iloc[-1]

    # Divergence: current price is near the low, but OBV has risen since
    price_now = close.iloc[-1]
    near_price_low = (price_now - price_low) / price_low * 100 < 5 if price_low > 0 else False

    # OBV has risen since the price low — bullish divergence
    obv_diverging = obv_now > obv_at_price_low * 1.02  # 2% buffer for noise

    passed = near_price_low and obv_diverging

    if passed:
        reason = "OBV Bullish Divergence detected"
    elif not near_price_low:
        reason = f"No OBV divergence — price {price_now:.2f} not near recent low {price_low:.2f}"
    else:
        reason = "No OBV divergence — OBV also declining"

    return FilterResult(
        passed=passed,
        reason=reason,
        details={
            "price_now": round(price_now, 2),
            "price_low": round(price_low, 2),
            "obv_now": round(obv_now, 0),
            "obv_at_low": round(obv_at_price_low, 0),
        },
    )


# ======================================================================
# CMF (Chaikin Money Flow) — Positive / Improving
# ======================================================================

def filter_cmf_positive(df: pd.DataFrame) -> FilterResult:
    """Check that CMF > threshold or improving over the last 20 days."""
    if "CMF" not in df.columns:
        return FilterResult(passed=False, reason="CMF not computed")

    if len(df) < 40:
        return FilterResult(passed=False, reason="Insufficient history for CMF check")

    cmf_now = df["CMF"].iloc[-1]
    cmf_20d_ago = df["CMF"].iloc[-20] if len(df) >= 20 else cmf_now

    # Pass if CMF > 0 now, OR CMF has improved by > 0.05 in 20 days
    passed = cmf_now > CMF_THRESHOLD or (cmf_now - cmf_20d_ago) > 0.05

    return FilterResult(
        passed=passed,
        reason=(
            f"CMF={cmf_now:.3f} {'>' if cmf_now > CMF_THRESHOLD else '<='} threshold, "
            f"20d change={cmf_now - cmf_20d_ago:+.3f}"
        ),
        details={"cmf": round(cmf_now, 4), "cmf_20d_change": round(cmf_now - cmf_20d_ago, 4)},
    )


# ======================================================================
# A/D Line — Positive Slope
# ======================================================================

def filter_ad_slope_positive(df: pd.DataFrame) -> FilterResult:
    """Check that A/D line slope is positive over the lookback period."""
    if "AD_Slope" not in df.columns:
        return FilterResult(passed=False, reason="AD Slope not computed")

    if len(df) < AD_SLOPE_LOOKBACK:
        return FilterResult(passed=False, reason="Insufficient history for AD slope")

    ad_slope = df["AD_Slope"].iloc[-1]
    passed = ad_slope > 0

    return FilterResult(
        passed=passed,
        reason=f"AD Slope={ad_slope:.6f} {'>' if passed else '<='} 0",
        details={"ad_slope": ad_slope},
    )


# ======================================================================
# Volatility Contraction
# ======================================================================

def filter_volatility_contraction(df: pd.DataFrame) -> FilterResult:
    """
    Detect volatility contraction — suggests "coiled spring" / accumulation:
    - ATR14 < ATR50 * 0.85 (ATR compressing)
    - Bollinger Band Width declining over last 60 days
    """
    details: dict[str, Any] = {}

    if len(df) < BB_WIDTH_COMPRESSION_LOOKBACK:
        return FilterResult(passed=False, reason="Insufficient history for volatility check")

    # ATR Compression
    atr_compressing = True
    if "ATR14" in df.columns and "ATR50" in df.columns:
        atr14 = df["ATR14"].iloc[-1]
        atr50 = df["ATR50"].iloc[-1]
        atr_compressing = atr14 < atr50 * 0.9  # slightly tighter than 0.85 for scoring
        details["atr14"] = round(atr14, 4)
        details["atr50"] = round(atr50, 4)
        details["atr_compressing"] = atr_compressing

    # BB Width declining
    bb_contracting = True
    if "BB_Width" in df.columns:
        bb_width = df["BB_Width"].dropna()
        if len(bb_width) >= BB_WIDTH_COMPRESSION_LOOKBACK:
            recent_bb = bb_width.iloc[-BB_WIDTH_COMPRESSION_LOOKBACK:]
            bb_contracting = recent_bb.iloc[-1] < recent_bb.iloc[0]
            details["bb_width_now"] = round(recent_bb.iloc[-1], 2)
            details["bb_width_30d_ago"] = round(recent_bb.iloc[0], 2)
            details["bb_contracting"] = bb_contracting

    passed = atr_compressing or bb_contracting

    return FilterResult(
        passed=passed,
        reason=(
            f"ATR compressing: {atr_compressing}, BB contracting: {bb_contracting}"
        ),
        details=details,
    )


# ======================================================================
# Market Cap Filter
# ======================================================================

def filter_min_market_cap(market_cap: float | None) -> FilterResult:
    """Reject if market cap is below MIN_MARKET_CAP."""
    if market_cap is None:
        return FilterResult(
            passed=False,
            reason="Market cap data unavailable",
            details={"market_cap": None},
        )
    passed = market_cap >= MIN_MARKET_CAP
    return FilterResult(
        passed=passed,
        reason=(
            f"MarketCap ${market_cap:,.0f} "
            f"{'>=' if passed else '<'} MIN_MARKET_CAP ${MIN_MARKET_CAP:,.0f}"
        ),
        details={"market_cap": market_cap},
    )


# ======================================================================
# Master filter runner
# ======================================================================

@dataclass
class AllFilterResults:
    """Aggregated results from all filters."""
    min_price: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    min_volume: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    min_market_cap: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    sufficient_history: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    bear_market: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    consolidation: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    volume_accumulation: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    obv_divergence: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    cmf_positive: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    ad_slope: FilterResult = field(default_factory=lambda: FilterResult(False, ""))
    volatility_contraction: FilterResult = field(default_factory=lambda: FilterResult(False, ""))

    def all_passed(self) -> bool:
        """Return True if all filters passed."""
        filters = [
            self.min_price, self.min_volume, self.min_market_cap, self.sufficient_history,
            self.bear_market, self.consolidation, self.volume_accumulation,
            self.obv_divergence, self.cmf_positive, self.ad_slope,
            self.volatility_contraction,
        ]
        return all(f.passed for f in filters)

    def passed_count(self) -> int:
        """How many filters passed."""
        filters = [
            self.min_price, self.min_volume, self.min_market_cap, self.sufficient_history,
            self.bear_market, self.consolidation, self.volume_accumulation,
            self.obv_divergence, self.cmf_positive, self.ad_slope,
            self.volatility_contraction,
        ]
        return sum(1 for f in filters if f.passed)


def run_all_filters(df: pd.DataFrame, market_cap: float | None = None) -> AllFilterResults:
    """
    Run every filter against *df*.

    Returns an AllFilterResults struct — call .all_passed() for the go/no-go.
    """
    return AllFilterResults(
        min_price=filter_min_price(df),
        min_volume=filter_min_volume(df),
        min_market_cap=filter_min_market_cap(market_cap),
        sufficient_history=filter_sufficient_history(df),
        bear_market=filter_bear_market(df),
        consolidation=filter_consolidation(df),
        volume_accumulation=filter_volume_accumulation(df),
        obv_divergence=filter_obv_divergence(df),
        cmf_positive=filter_cmf_positive(df),
        ad_slope=filter_ad_slope_positive(df),
        volatility_contraction=filter_volatility_contraction(df),
    )
