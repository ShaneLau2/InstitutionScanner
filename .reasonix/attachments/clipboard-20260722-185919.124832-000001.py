"""
downloader.py — Multi-source ticker universe & historical data manager.

Responsible for:
1. Building the ticker universe (stocks: NASDAQ / NYSE / AMEX; ETFs: free lists).
2. Downloading 10+ years of daily OHLCV data via yfinance.
3. Incremental cache: existing data is extended, new data is downloaded fresh.
4. Rate-limiting, batch parallelism (ThreadPoolExecutor), and graceful error recovery.
"""

from __future__ import annotations

import json
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from config import (
    CACHE_DIR,
    DOWNLOAD_RATE_LIMIT_PAUSE,
    DOWNLOAD_RETRIES,
    DOWNLOAD_THREADS,
    DOWNLOAD_TIMEOUT,
    HISTORY_YEARS,
    LOG_DIR,
    MAX_DOWNLOAD_ERRORS,
    MIN_MARKET_CAP,
    MIN_PRICE,
    MIN_VOLUME,
)

logger = logging.getLogger("institution_scanner.downloader")
logger.setLevel(logging.DEBUG)

# Attach a rotating file handler so we don't lose logs
_fh = logging.FileHandler(LOG_DIR / "downloader.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TickerInfo:
    """Minimal metadata for a single ticker."""
    ticker: str
    name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    is_etf: bool = False


# ---------------------------------------------------------------------------
# Ticker universe builders
# ---------------------------------------------------------------------------

# Known ETF tickers from major issuers (free — no API key required).
# These are widely-traded, high-volume ETFs.  We augment with dynamic lists below.
_STATIC_ETFS: set[str] = {
    # US Equity
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "MDY", "VO", "VB",
    "IWF", "IWD", "IWB", "RSP", "SCHX", "SCHB", "ITOT",
    # Sectors
    "XLF", "XLK", "XLY", "XLP", "XLE", "XLV", "XLI", "XLB", "XLRE", "XLU",
    "VGT", "VFH", "VHT", "VIS", "VAW", "VNQ", "VPU", "VDE", "VDC", "VOX",
    # Bonds
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "BIL",
    "MUB", "VCIT", "VCSH", "VGLT", "VGIT", "VGSH", "BSV",
    # International
    "EFA", "EEM", "VXUS", "VEA", "VWO", "IEMG", "EWJ", "EWG", "EWU",
    "FXI", "KWEB", "ASHR", "INDA", "EWZ", "EWY", "EZA", "RSX",
    # Commodities
    "GLD", "IAU", "SLV", "USO", "UNG", "DBC", "DBA", "PDBC",
    # Inverse / Leveraged (common institutional hedges)
    "SQQQ", "TQQQ", "SOXS", "SOXL", "UVXY", "SVXY", "LABD", "LABU",
    "TMF", "TMV", "UUP", "UDOW", "SDOW",
    # Smart Beta / Thematic
    "MTUM", "VLUE", "QUAL", "SIZE", "USMV", "HDV", "VYM", "SCHD", "DGRO",
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "ICLN", "TAN", "LIT",
    # VIX
    "VXX", "VIXY",
    # Crypto
    "GBTC", "ETHE",
    # Real Estate
    "IYR", "SCHH",
    # Healthcare
    "IBB", "XBI",
    # Tech
    "IGV", "SOXX", "SMH", "SKYY", "CIBR", "HACK",
    # Other
    "XRT", "KRE", "XHB", "XME", "XOP",
}

# ---- Ticker validation (no regex — simple rules) ----

_INVALID_SUFFIXES: set[str] = {
    "W", "R", "P", "Z",    # warrants, rights, preferred, misc
}
_INVALID_CHARS: set[str] = {"=", "$", "^", ".", "+", "-"}

_REJECTED_EXCHANGES: set[str] = {
    "OTC", "OTC BB", "OTCQB", "PINX", "GREY",
}


def _is_viable_ticker(symbol: str, exchange: str = "") -> bool:
    """Return True if the ticker looks like a vanilla common stock / ETF.

    Rejects anything with:
    - Special chars: = $ ^ . + -   (AAC=, ALUB+, BRK.B)
    - Length > 5                       (ESLAW, FACWW, FBYDP — warrants/SPACs)
    - Trailing W/R/P/Z                 (warrants, rights, preferred)
      *unless* the whole symbol is ≤3 chars (e.g. CAT — legit names)
    - OTC / Pink Sheets exchanges
    """
    if not symbol or len(symbol) > 5:
        return False
    for ch in symbol:
        if ch in _INVALID_CHARS:
            return False
    if len(symbol) >= 4 and symbol[-1].upper() in _INVALID_SUFFIXES:
        return False
    if exchange and exchange.upper() in _REJECTED_EXCHANGES:
        return False
    return True


def _fetch_nasdaq_traded() -> list[TickerInfo]:
    """
    Download the official NASDAQ Traded symbols file (free, no API key).
    Returns a list of TickerInfo objects for all listed securities.
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    tickers: list[TickerInfo] = []
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # The file is pipe-delimited, last line is a footer
        reader = csv.DictReader(
            StringIO(resp.text),
            delimiter="|",
        )
        skipped = 0
        for row in reader:
            symbol = (row.get("NASDAQ Symbol") or "").strip()
            if not symbol or symbol == "File Creation Time":
                continue
            # Skip test issues
            if row.get("Test Issue", "N") == "Y":
                continue
            exchange = (row.get("Listing Exchange") or "").strip()
            if not _is_viable_ticker(symbol, exchange):
                skipped += 1
                continue
            name = (row.get("Security Name") or "").strip()
            is_etf = row.get("ETF", "N") == "Y"
            tickers.append(TickerInfo(
                ticker=symbol,
                name=name,
                exchange=exchange,
                is_etf=is_etf,
            ))
        logger.info(
            "Fetched %d tickers from NASDAQ traded list (%d filtered).",
            len(tickers), skipped,
        )
    except Exception as exc:
        logger.warning("Could not fetch NASDAQ traded list: %s", exc)
    return tickers


def _fetch_wikipedia_sp500() -> list[TickerInfo]:
    """Scrape S&P 500 constituents from Wikipedia (free, no auth)."""
    tickers: list[TickerInfo] = []
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        for _, row in df.iterrows():
            symbol = str(row.get("Symbol", "")).replace(".", "-").strip()
            name = str(row.get("Security", "")).strip()
            sector = str(row.get("GICS Sector", "")).strip()
            industry = str(row.get("GICS Sub-Industry", "")).strip()
            if symbol:
                tickers.append(TickerInfo(
                    ticker=symbol,
                    name=name,
                    exchange="NYSE/NASDAQ",
                    sector=sector,
                    industry=industry,
                ))
        logger.info("Fetched %d S&P 500 tickers from Wikipedia.", len(tickers))
    except Exception as exc:
        logger.warning("Could not fetch S&P 500 from Wikipedia: %s", exc)
    return tickers


def _fetch_etf_list() -> list[TickerInfo]:
    """
    Build an ETF ticker list from multiple free sources:
    - NASDAQ traded list (ETF flag)
    - Static curated list of major ETFs
    - ETFdb screener (free HTML table)
    """
    etfs: dict[str, TickerInfo] = {}

    # 1. NASDAQ traded (ETF = Y)
    for ti in _fetch_nasdaq_traded():
        if ti.is_etf:
            etfs[ti.ticker] = ti

    # 2. Static curated list (covers many that NASDAQ list might miss)
    for sym in _STATIC_ETFS:
        if sym not in etfs:
            etfs[sym] = TickerInfo(ticker=sym, is_etf=True)

    # 3. ETF.com screener table (free)
    try:
        url = "https://etfdb.com/screener/"
        tables = pd.read_html(url)
        # The main screener table is usually the largest
        for table in tables:
            if "Symbol" in table.columns:
                for _, row in table.iterrows():
                    sym = str(row.get("Symbol", "")).strip()
                    name = str(row.get("ETF Name", "")).strip()
                    if sym and sym not in etfs:
                        etfs[sym] = TickerInfo(ticker=sym, name=name, is_etf=True)
                break
        logger.info("ETFdb screener yielded %d unique ETFs.", len(etfs))
    except Exception as exc:
        logger.warning("Could not fetch ETFdb screener: %s", exc)

    return list(etfs.values())


def build_ticker_universe(
    include_stocks: bool = True,
    include_etfs: bool = True,
) -> tuple[list[TickerInfo], list[TickerInfo]]:
    """
    Build the complete ticker universe.

    Returns:
        (stocks, etfs) — two lists of TickerInfo.
        Each ticker is deduplicated by symbol.
    """
    stocks: dict[str, TickerInfo] = {}
    etfs: dict[str, TickerInfo] = {}

    if include_stocks:
        # NASDAQ traded
        for ti in _fetch_nasdaq_traded():
            if not ti.is_etf:
                key = ti.ticker.upper()
                if key not in stocks:
                    stocks[key] = ti

        # S&P 500 (supplements sector/industry data)
        for ti in _fetch_wikipedia_sp500():
            key = ti.ticker.upper()
            if key in stocks:
                # Merge metadata
                existing = stocks[key]
                if ti.sector and not existing.sector:
                    existing.sector = ti.sector
                if ti.industry and not existing.industry:
                    existing.industry = ti.industry
            else:
                stocks[key] = ti

    if include_etfs:
        for ti in _fetch_etf_list():
            key = ti.ticker.upper()
            if key not in etfs:
                etfs[key] = ti

    stock_list = sorted(stocks.values(), key=lambda x: x.ticker)
    etf_list = sorted(etfs.values(), key=lambda x: x.ticker)

    logger.info(
        "Universe built: %d stocks, %d ETFs",
        len(stock_list), len(etf_list),
    )
    return stock_list, etf_list


# ---------------------------------------------------------------------------
# Data cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    """File path for a ticker's cached CSV."""
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.csv"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    """Load cached OHLCV data for a ticker, or None if not found / corrupted."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        # Ensure required columns
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                logger.warning("Cache for %s missing column %s — ignoring.", ticker, col)
                return None
        return df
    except Exception:
        logger.warning("Corrupted cache for %s — will re-download.", ticker)
        return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    """Persist OHLCV data to CSV."""
    path = _cache_path(ticker)
    df.to_csv(path)


# ---------------------------------------------------------------------------
# Metadata cache (market cap, etc.)
# ---------------------------------------------------------------------------

def _meta_path(ticker: str) -> Path:
    """File path for a ticker's cached metadata JSON."""
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.json"


def _save_meta(ticker: str, data: dict) -> None:
    """Persist metadata (marketCap, etc.) to JSON."""
    _meta_path(ticker).write_text(json.dumps(data, default=str))


def _load_meta(ticker: str) -> dict | None:
    """Load cached metadata, or None."""
    path = _meta_path(ticker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fetch_market_cap_from_yf(ticker: str) -> float | None:
    """
    Fetch market cap from yfinance Ticker.info for a single ticker.

    Returns a float in USD or None on failure.
    """
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        mc = info.get("marketCap")
        if mc is not None and isinstance(mc, (int, float)) and mc > 0:
            return float(mc)
        return None
    except Exception:
        return None


def get_market_cap(ticker: str) -> float | None:
    """
    Return the cached market cap for *ticker*.

    If no cached metadata exists, attempts a live fetch from yfinance,
    caches the result, and returns it.  Returns None when unavailable.
    """
    meta = _load_meta(ticker)
    if meta and "marketCap" in meta:
        return float(meta["marketCap"])

    # Try live fetch
    mc = _fetch_market_cap_from_yf(ticker)
    if mc is not None:
        _save_meta(ticker, {"marketCap": mc, "fetchedAt": datetime.now().isoformat()})
        return mc

    return None


def _download_single(ticker: str) -> pd.DataFrame | None:
    """
    Download full history for *ticker* from yfinance.
    Returns a DataFrame or None on failure.
    """
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            tkr = yf.Ticker(ticker)
            # Request the full period directly — skip info() to avoid extra HTTP calls
            end_date = datetime.now()
            start_date = end_date - timedelta(days=HISTORY_YEARS * 365 + 30)
            df = tkr.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                timeout=DOWNLOAD_TIMEOUT,
            )
            if df is None or df.empty:
                return None
            # Standardise columns
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })
            # Keep only the standard OHLCV columns
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            # Drop rows where Close is NaN
            df = df.dropna(subset=["Close"])
            if df.empty:
                return None
            return df
        except Exception as exc:
            msg = str(exc).lower()
            # 404 / delisted / timeout / curl errors — skip instantly
            if any(kw in msg for kw in ("404", "not found", "delisted", "no timezone", "timeout", "timed out", "no data found", "failed to perform", "curl")):
                return None
            # 401 / 429 rate limits — back off harder
            if "401" in msg or "429" in msg or "rate limit" in msg:
                delay = 5 + (attempt * 5)
                logger.debug(
                    "Rate-limited on %s (attempt %d/%d), backing off %ds...",
                    ticker, attempt, DOWNLOAD_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            logger.debug("Attempt %d/%d failed for %s: %s", attempt, DOWNLOAD_RETRIES, ticker, exc)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2 ** attempt)
    return None


def download_ticker(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """
    Get OHLCV data for *ticker*.
    - If cached data exists, load it and download only the missing tail.
    - If *force* is True, re-download everything.
    """
    if force:
        df = _download_single(ticker)
        if df is not None:
            _save_cache(ticker, df)
        return df

    cached = _load_cache(ticker)
    if cached is None:
        # No cache — full download
        df = _download_single(ticker)
        if df is not None:
            _save_cache(ticker, df)
        return df

    # Incremental update: download only from the last cached date
    last_date = cached.index.max()
    today = datetime.now()

    # Normalise to naive datetime for comparison (Yahoo data may be tz-aware)
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.to_pydatetime()
    if last_date.tzinfo is not None:
        last_date = last_date.replace(tzinfo=None)

    # Guard: if last_date is somehow in the future, skip update
    if (today - last_date).days <= 1:
        return cached  # already up-to-date

    try:
        tkr = yf.Ticker(ticker)
        # Use date strings to avoid tz mismatch inside yfinance
        start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")
        new_df = tkr.history(
            start=start_str,
            end=end_str,
            interval="1d",
            auto_adjust=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
        if new_df is not None and not new_df.empty:
            new_df = new_df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            new_df = new_df[["Open", "High", "Low", "Close", "Volume"]]
            new_df = new_df.dropna(subset=["Close"])
            # Strip timezone from new data to match cached
            if new_df.index.tz is not None:
                new_df.index = new_df.index.tz_localize(None)
            if not new_df.empty:
                combined = pd.concat([cached, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                _save_cache(ticker, combined)
                return combined
    except Exception as exc:
        logger.debug("Incremental update failed for %s: %s — using cache as-is.", ticker, exc)

    return cached


def download_batch(
    tickers: list[TickerInfo],
    desc: str = "Downloading",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download data for a list of tickers using ThreadPoolExecutor.

    All tickers are submitted at once — the pool's max_workers threads
    pull continuously from the queue with no gaps between batches.

    Args:
        tickers: List of TickerInfo.
        desc: Progress bar label.
        force: If True, ignore cache and re-download everything.

    Returns:
        {ticker: DataFrame} mapping (only successful downloads).
    """
    results: dict[str, pd.DataFrame] = {}
    symbols = [t.ticker for t in tickers]

    total = len(symbols)
    skipped_delisted = 0

    # Single-threaded download with inter-request pause (respects Yahoo's
    # ~60 req/min soft limit).  Parallel path kept for DOWNLOAD_THREADS > 1.
    if DOWNLOAD_THREADS <= 1:
        for sym in tqdm(symbols, desc=desc, unit="ticker"):
            try:
                df = download_ticker(sym, force=force)
                if df is not None and not df.empty:
                    results[sym] = df
                else:
                    skipped_delisted += 1
            except Exception:
                skipped_delisted += 1
            time.sleep(DOWNLOAD_RATE_LIMIT_PAUSE)
    else:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            futures: dict[Any, str] = {
                pool.submit(download_ticker, sym, force): sym for sym in symbols
            }

            for future in tqdm(
                as_completed(futures),
                total=total,
                desc=desc,
                unit="ticker",
            ):
                sym = futures[future]
                try:
                    df = future.result(timeout=DOWNLOAD_TIMEOUT + 10)
                    if df is not None and not df.empty:
                        results[sym] = df
                    else:
                        skipped_delisted += 1
                except Exception as exc:
                    logger.debug("Download exception for %s: %s", sym, exc)
                    skipped_delisted += 1

    logger.info(
        "Download batch complete: %d/%d tickers succeeded, %d delisted/no-data skipped.",
        len(results), total, skipped_delisted,
    )
    return results


def get_etf_fund_flows(ticker: str) -> float | None:
    """
    Attempt to retrieve ETF fund flow data from free sources.

    Currently uses yfinance info dict which sometimes contains
    'fundFamily', 'netAssets', etc. — not daily flows.
    For daily flows a paid API (e.g. ETFdb Pro, Bloomberg) is needed,
    so this function returns None when flows are unavailable.

    Returns:
        Estimated net flow (positive = inflow) or None.
    """
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        # Some tickers expose fund flow data via yfinance's info dict
        net_assets = info.get("netAssets")
        if net_assets:
            return float(net_assets)
        return None
    except Exception:
        return None
