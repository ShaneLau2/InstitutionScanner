"""
InstitutionScanner — config.py

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


# ======================================================================
# Ticker & Market Filters
# ======================================================================
MIN_PRICE: float = 3.0        # Minimum close price (USD) — ignore penny stocks
MAX_PRICE: float = 500.0      # Maximum close price
MIN_VOLUME: int = 100_000     # Minimum daily volume (shares)
MIN_MARKET_CAP: float = 100e6  # Minimum market cap (USD) — ignore nano-caps

# ======================================================================
# Data Download
# ======================================================================
HISTORY_YEARS: int = 10                  # Years of daily data to pull
DOWNLOAD_THREADS: int = 12               # concurrent.futures workers
DOWNLOAD_BATCH_SIZE: int = 50            # tickers per batch
DOWNLOAD_RATE_LIMIT_PAUSE: float = 0.05  # seconds between tickers inside a batch
DOWNLOAD_BATCH_PAUSE: float = 1.0        # seconds between batches
DOWNLOAD_RETRIES: int = 3                # retries on transient errors
DOWNLOAD_TIMEOUT: int = 30               # seconds per ticker
MAX_DOWNLOAD_ERRORS: int = 500           # abort if this many consecutive errors (harmless 404s from delisted tickers are common)

# Ticker list sources (free, no API key required)
TICKER_SOURCES: list[str] = field(default_factory=lambda: [
    # NASDAQ official FTP lists
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
    # Alternative free sources (used as fallback)
])

# ETF list sources
ETF_SOURCES: list[str] = field(default_factory=lambda: [
    # Common free ETF lists
])

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
BEAR_DECLINE_PCT: float = -30.0       # Min decline over lookback period
BEAR_LOOKBACK_YEARS: int = 2           # Years for decline calculation
BEAR_MA200_DECLINING_DAYS: int = 60    # MA200 must be declining for at least N days

# Bottom consolidation
CONSOLIDATION_DAYS: int = 60           # Lookback for consolidation check
CONSOLIDATION_MAX_RANGE_PCT: float = 15.0  # Max % range during consolidation

# Volume accumulation
VOLUME_ACCUM_RATIO: float = 1.4        # Vol MA20 > Vol MA120 * ratio
VOLUME_ACCUM_MIN_DAYS: int = 30        # Must persist for this many consecutive days

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
SCAN_THREADS: int = 8           # Threads for parallel indicator calculation
CHECKPOINT_INTERVAL: int = 100  # Save checkpoint every N tickers
ENABLE_CHECKPOINT: bool = True

# ETF Fund Flows (optional, requires a free source)
ENABLE_FUND_FLOWS: bool = True

# Volume Profile in scoring
ENABLE_VOLUME_PROFILE: bool = True
