"""
ScannerGui — config.py

Central configuration for the Institutional Accumulation Scanner.
All tunable parameters live here so no magic numbers appear in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
CACHE_DIR: Final[Path] = BASE_DIR / "cache"
OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
LOG_DIR: Final[Path] = BASE_DIR / "logs"

# Ensure directories exist
for _d in (CACHE_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Cache subdirectories per market
for _sub in ("a_share", "us"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)


# ======================================================================
# Ticker & Market Filters
# ======================================================================
MIN_PRICE: float = 5.0        # Minimum close price (CNY) — ignore penny stocks
MAX_PRICE: float = 800.0      # Maximum close price for A-shares
MIN_VOLUME: int = 200_000     # Minimum daily volume (shares)
MIN_MARKET_CAP: float = 1e8    # Minimum market cap (CNY) — ignore micro-caps
# Deprecated — use A_SHARE_CONFIG.excluded_keywords instead.
# Kept here for backward compatibility with imports.
EXCLUDED_SECURITY_KEYWORDS = property(lambda self: A_SHARE_CONFIG.excluded_keywords)
_EXCLUDED_SECURITY_KEYWORDS: tuple[str, ...] = (
    "债",
    "货币",
    "同业存单",
    "短融",
    "中票",
    "REIT",
    "浙商沪",
)
# Legacy alias for module-level imports (downloader._is_excluded_security_name)
EXCLUDED_SECURITY_KEYWORDS = _EXCLUDED_SECURITY_KEYWORDS
# ======================================================================
# Market-specific configurations (A-share vs US stocks)
# ======================================================================

@dataclass(frozen=True)
class MarketConfig:
    """Market-specific thresholds and filters."""
    label: str
    currency: str
    min_price: float = 5.0
    max_price: float = 800.0
    min_volume: int = 200_000
    min_market_cap: float = 1e8
    excluded_keywords: tuple[str, ...] = ()
    bear_decline_pct: float = -20.0
    download_threads: int = 2


A_SHARE_CONFIG: Final[MarketConfig] = MarketConfig(
    label="A股", currency="元", min_price=5.0, max_price=800.0,
    min_volume=200_000, min_market_cap=1e8,
    excluded_keywords=("债", "货币", "同业存单", "短融", "中票", "REIT", "浙商沪"),
    bear_decline_pct=-20.0, download_threads=2,
)

US_STOCK_CONFIG: Final[MarketConfig] = MarketConfig(
    label="美股", currency="USD", min_price=5.0, max_price=5000.0,
    min_volume=200_000, min_market_cap=1e8,
    excluded_keywords=(),
    bear_decline_pct=-30.0, download_threads=2,
)

# Shortcut for backward compatibility
MARKET_CONFIGS: Final[dict[str, MarketConfig]] = {
    "a_share": A_SHARE_CONFIG,
    "us": US_STOCK_CONFIG,
}

# ======================================================================
# Data Download
# ======================================================================
HISTORY_YEARS: int = 10                  # Years of daily data to pull

# Yahoo Finance rate limits (empirically observed, not officially documented):
#   ~1-2 req/s  per IP without TLS fingerprint evasion
#   ~60 req/min Yahoo's unofficial guideline
#   ~900 burst requests before hard throttling
#   Historical data (period > 1y) is heavier — triggers limits faster
#   TLS fingerprint: Python's 'requests' is detected as bot since 2025
#
# Strategy: 2 threads, 1s pause = ~2 req/s = ~120 req/min
# Phase 2 (indicator computation) stays parallel since it's CPU-bound, no network
DOWNLOAD_THREADS: int = 2
DOWNLOAD_RATE_LIMIT_PAUSE: float = 1.0
DOWNLOAD_RETRIES: int = 2                # retries on transient errors (401s, 429s, timeouts) — don't waste time retrying dead URLs
DOWNLOAD_TIMEOUT: int = 10               # seconds per ticker (lower = less accumulated delay on dead URLs)
MAX_DOWNLOAD_ERRORS: int = 2000          # abort if this many consecutive errors (harmless 404s from delisted tickers are common)

# ======================================================================
# Indicator Parameters
# ======================================================================
MA_PERIODS: tuple[int, ...] = (20, 50, 100, 200)
EMA_PERIODS: tuple[int, ...] = (20, 50, 200)
ATR_PERIODS: tuple[int, ...] = (14, 50)
ADX_PERIOD: int = 14
CCI_PERIOD: int = 20
ROC_PERIOD: int = 21

VOLUME_MA_PERIODS: tuple[int, ...] = (20, 60, 120)
VOLUME_RATIO_PERIODS: tuple[int, ...] = (20, 60)
VOLUME_ZSCORE_PERIOD: int = 60
VOLUME_TREND_PERIOD: int = 60

OBV_SLOPE_PERIOD: int = 20
AD_SLOPE_PERIOD: int = 20
CMF_PERIOD: int = 21
MFI_PERIOD: int = 14
VWAP_PERIOD: int = 252

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
RSI_PERIODS: tuple[int, ...] = (14, 21)

HV_PERIODS: tuple[int, ...] = (20, 60)
BB_PERIOD: int = 20
BB_STD: float = 2.0
DONCHIAN_PERIOD: int = 20

REGRESSION_PERIOD: int = 60

# Volume Profile
VOLUME_PROFILE_BINS: int = 50
VOLUME_PROFILE_LOOKBACK: int = 252

# ======================================================================
# Filter Thresholds
# ======================================================================

# Long-term bear market
BEAR_DECLINE_PCT: float = -20.0       # Min decline over lookback period for A-shares
BEAR_LOOKBACK_YEARS: int = 2           # Years for decline calculation
BEAR_MA200_DECLINING_DAYS: int = 40    # MA200 must be declining for at least N days

# Bottom consolidation
CONSOLIDATION_DAYS: int = 45           # Lookback for consolidation check
CONSOLIDATION_MAX_RANGE_PCT: float = 20.0  # Max % range during consolidation

# Volume accumulation
VOLUME_ACCUM_RATIO: float = 1.2        # Vol MA20 > Vol MA120 * ratio
VOLUME_ACCUM_MIN_DAYS: int = 20        # Must persist for this many consecutive days

# OBV Bullish Divergence
OBV_DIVERGENCE_LOOKBACK: int = 60      # Days to check for price low vs OBV low

# CMF
CMF_THRESHOLD: float = 0.0             # CMF must exceed this

# AD Line
AD_SLOPE_LOOKBACK: int = 30            # AD slope must be positive over N days

# Volatility Contraction
ATR_COMPRESSION_LOOKBACK: int = 60     # ATR must decline over this many days
BB_WIDTH_COMPRESSION_LOOKBACK: int = 60

# ======================================================================
# Scoring Weights (total = 100)
# ======================================================================
@dataclass(frozen=True)
class ScoringWeights:
    trend: float = 20.0
    volume: float = 25.0
    accumulation: float = 25.0
    volatility: float = 15.0
    structure: float = 15.0

SCORING_WEIGHTS: Final[ScoringWeights] = ScoringWeights()

# ======================================================================
# Output
# ======================================================================
TOP_N_REPORT: int = 50
TOP_N_PARQUET: int = 200

# ======================================================================
# Runtime
# ======================================================================
SCAN_THREADS: int = 12          # Threads for parallel indicator calculation (numpy releases GIL)
CHECKPOINT_INTERVAL: int = 100  # Save checkpoint every N tickers
ENABLE_CHECKPOINT: bool = True

# ETF Fund Flows (optional, requires a free source)
ENABLE_FUND_FLOWS: bool = True

# Volume Profile in scoring
ENABLE_VOLUME_PROFILE: bool = True
