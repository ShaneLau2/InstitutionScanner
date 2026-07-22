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
    MARKET_CONFIGS,
    MIN_MARKET_CAP,
    MIN_PRICE,
    MAX_PRICE,
    MIN_VOLUME,
    OBV_DIVERGENCE_LOOKBACK,
    VOLUME_ACCUM_MIN_DAYS,
    VOLUME_ACCUM_RATIO,
)

logger = logging.getLogger("scanner_gui.filters")


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

def filter_min_price(df: pd.DataFrame, min_price: float = MIN_PRICE, max_price: float = MAX_PRICE) -> FilterResult:
    """Reject if latest close is outside the configured price range."""
    close = df["Close"].iloc[-1]
    passed = min_price <= close <= max_price
    return FilterResult(
        passed=passed,
        reason=f"收盘价 {close:.2f}，要求范围 {min_price:.2f}-{max_price:.2f}",
        details={"close": close},
    )


def filter_min_volume(df: pd.DataFrame, min_volume: int = MIN_VOLUME) -> FilterResult:
    """Reject if average daily volume (60d) is below min_volume."""
    vol_avg = df["Volume"].rolling(60, min_periods=30).mean().iloc[-1]
    passed = vol_avg >= min_volume
    return FilterResult(
        passed=passed,
        reason=f"AvgVol {vol_avg:,.0f} {'>=' if passed else '<'} {min_volume:,}",
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
    condition = vol_ma20 >= vol_ma120 * VOLUME_ACCUM_RATIO
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

    # Build consecutive-count series (forward walk — also gives trailing count)
    volumes = condition.reset_index(drop=True)
    consecutive_series = pd.Series(0, index=condition.index, dtype=int)
    cnt = 0
    for i in range(len(condition)):
        if condition.iloc[i]:
            cnt += 1
        else:
            cnt = 0
        consecutive_series.iloc[i] = cnt
    trailing_consecutive = int(consecutive_series.iloc[-1])
    max_consecutive = int(consecutive_series.max())
    passed = trailing_consecutive >= VOLUME_ACCUM_MIN_DAYS
    df["_VolAccumDays"] = consecutive_series  # temporary column for scoring

    return FilterResult(
        passed=passed,
        reason=(
            f"Volume accumulation: {trailing_consecutive} consecutive days "
            f"(need {VOLUME_ACCUM_MIN_DAYS}), "
            f"peak {max_consecutive}"
        ),
        details={
            "consecutive_days": trailing_consecutive,
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
    recent = df.iloc[-lookback:].copy()
    recent = recent[["Close", "OBV"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(recent) < max(20, lookback // 2):
        return FilterResult(passed=False, reason="Insufficient valid data for OBV divergence")

    close = recent["Close"]
    obv = recent["OBV"]
    split = max(len(recent) // 2, 1)
    first_half = recent.iloc[:split]
    second_half = recent.iloc[split:]
    if second_half.empty:
        return FilterResult(passed=False, reason="Insufficient valid data for OBV divergence")

    price_low_first = float(first_half["Close"].min())
    price_low_second = float(second_half["Close"].min())
    obv_low_first = float(first_half["OBV"].min())
    obv_low_second = float(second_half["OBV"].min())
    price_now = float(close.iloc[-1])
    obv_now = float(obv.iloc[-1])
    near_price_low = price_low_second > 0 and (price_now - price_low_second) / price_low_second < 0.05
    price_lower_low = price_low_second <= price_low_first * 1.02
    obv_higher_low = obv_low_second > obv_low_first
    obv_recovering = obv_now >= obv_low_second
    passed = near_price_low and price_lower_low and obv_higher_low and obv_recovering

    if passed:
        reason = "OBV Bullish Divergence detected"
    elif not near_price_low:
        reason = f"No OBV divergence — price {price_now:.2f} not near recent low {price_low_second:.2f}"
    else:
        reason = "No OBV divergence — price/OBV lows do not confirm accumulation"

    return FilterResult(
        passed=passed,
        reason=reason,
        details={
            "price_now": round(price_now, 2),
            "price_low": round(price_low_second, 2),
            "prior_price_low": round(price_low_first, 2),
            "obv_now": round(obv_now, 0),
            "obv_low": round(obv_low_second, 0),
            "prior_obv_low": round(obv_low_first, 0),
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
    if pd.isna(cmf_now) or pd.isna(cmf_20d_ago):
        return FilterResult(passed=False, reason="CMF data unavailable")

    cmf_change = cmf_now - cmf_20d_ago
    passed = cmf_now > CMF_THRESHOLD or cmf_change > 0.05

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

    if len(df) < max(BB_WIDTH_COMPRESSION_LOOKBACK, ATR_COMPRESSION_LOOKBACK):
        return FilterResult(passed=False, reason="Insufficient history for volatility check")

    atr_compressing = False
    if "ATR14" in df.columns:
        atr_values = df["ATR14"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(atr_values) >= ATR_COMPRESSION_LOOKBACK:
            atr_now = atr_values.iloc[-1]
            atr_start = atr_values.iloc[-ATR_COMPRESSION_LOOKBACK]
            atr_compressing = pd.notna(atr_now) and pd.notna(atr_start) and atr_now < atr_start
            details["atr14"] = round(atr_now, 4)
            details["atr14_lookback"] = round(atr_start, 4)
            details["atr_compressing"] = atr_compressing

    # BB Width declining
    bb_contracting = False
    if "BB_Width" in df.columns:
        bb_width = df["BB_Width"].dropna()
        if len(bb_width) >= BB_WIDTH_COMPRESSION_LOOKBACK:
            recent_bb = bb_width.iloc[-BB_WIDTH_COMPRESSION_LOOKBACK:]
            bb_contracting = recent_bb.iloc[-1] < recent_bb.iloc[0]
            details["bb_width_now"] = round(recent_bb.iloc[-1], 2)
            details["bb_width_lookback"] = round(recent_bb.iloc[0], 2)
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

def filter_min_market_cap(market_cap: float | None, required: bool = True, min_market_cap: float = MIN_MARKET_CAP) -> FilterResult:
    """Reject if market cap is below min_market_cap."""
    if market_cap is None:
        return FilterResult(
            passed=not required,
            reason="市值数据不可用" if required else "ETF不要求市值数据",
            details={"market_cap": None},
        )
    passed = market_cap >= min_market_cap
    return FilterResult(
        passed=passed,
        reason=(
            f"市值 {market_cap:,.0f} "
            f"{'>=' if passed else '<'} 最低市值 {min_market_cap:,.0f}"
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
        """Return True when basic data checks pass and enough signals agree."""
        base_filters = [self.min_price, self.min_volume, self.min_market_cap, self.sufficient_history]
        signal_filters = [
            self.bear_market, self.consolidation, self.volume_accumulation,
            self.obv_divergence, self.cmf_positive, self.ad_slope,
            self.volatility_contraction,
        ]
        primary_accumulation = [
            self.volume_accumulation,
            self.obv_divergence,
            self.cmf_positive,
            self.ad_slope,
        ]
        structure_signals = [self.consolidation, self.volatility_contraction]
        signal_count = sum(1 for item in signal_filters if item.passed)
        has_accumulation = any(item.passed for item in primary_accumulation)
        has_structure = any(item.passed for item in structure_signals)
        return (
            all(item.passed for item in base_filters)
            and signal_count >= 4
            and has_accumulation
            and has_structure
        )

    def signal_count(self) -> int:
        return sum(1 for item in (
            self.bear_market, self.consolidation, self.volume_accumulation,
            self.obv_divergence, self.cmf_positive, self.ad_slope,
            self.volatility_contraction,
        ) if item.passed)

    def passed_count(self) -> int:
        filters = [
            self.min_price, self.min_volume, self.min_market_cap, self.sufficient_history,
            self.bear_market, self.consolidation, self.volume_accumulation,
            self.obv_divergence, self.cmf_positive, self.ad_slope,
            self.volatility_contraction,
        ]
        return sum(1 for item in filters if item.passed)

    def all_filter_count(self) -> int:
        """Alias for passed_count — kept for backward compatibility."""
        return self.passed_count()


def run_all_filters(
    df: pd.DataFrame,
    market_cap: float | None = None,
    require_market_cap: bool = True,
    market: str = "a_share",
) -> AllFilterResults:
    """
    Run every filter against *df*.

    Uses market-specific thresholds from MarketConfig when available.

    Returns an AllFilterResults struct — call .all_passed() for the go/no-go.
    """
    cfg = MARKET_CONFIGS.get(market)
    return AllFilterResults(
        min_price=filter_min_price(df, cfg.min_price, cfg.max_price) if cfg else filter_min_price(df),
        min_volume=filter_min_volume(df, cfg.min_volume) if cfg else filter_min_volume(df),
        min_market_cap=filter_min_market_cap(market_cap, required=require_market_cap,
                                              min_market_cap=cfg.min_market_cap if cfg else MIN_MARKET_CAP),
        sufficient_history=filter_sufficient_history(df),
        bear_market=filter_bear_market(df),
        consolidation=filter_consolidation(df),
        volume_accumulation=filter_volume_accumulation(df),
        obv_divergence=filter_obv_divergence(df),
        cmf_positive=filter_cmf_positive(df),
        ad_slope=filter_ad_slope_positive(df),
        volatility_contraction=filter_volatility_contraction(df),
    )
