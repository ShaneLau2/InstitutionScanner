"""
scanner.py — Orchestration engine for the Institutional Accumulation Scanner.

Responsibilities:
1. Load the ticker universe (stocks + ETFs).
2. Download / update cached OHLCV data in parallel.
3. Compute all indicators for each ticker.
4. Run screening filters.
5. Score passing tickers with the accumulation scoring system.
6. Rank and return results.

Supports checkpointing: if the scan is interrupted, resume from the last
saved checkpoint instead of restarting.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CACHE_DIR,
    CHECKPOINT_INTERVAL,
    DOWNLOAD_THREADS,
    ENABLE_CHECKPOINT,
    LOG_DIR,
    OUTPUT_DIR,
    SCAN_THREADS,
    TOP_N_PARQUET,
)
from downloader import (
    TickerInfo,
    _load_cache,
    build_ticker_universe,
    download_batch,
    download_ticker,
    get_etf_fund_flows,
)
from indicators import compute_all_indicators
from filters import run_all_filters
from score import score_ticker, ScoreBreakdown

logger = logging.getLogger("institution_scanner.scanner")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_DIR / "scanner.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)


# ======================================================================
# Scan result container
# ======================================================================

@dataclass
class ScanResult:
    """Full result for one scanned ticker."""
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    is_etf: bool = False
    close: float = 0.0
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    obv: float = np.nan
    cmf: float = np.nan
    ad: float = np.nan
    atr14: float = np.nan
    rsi14: float = np.nan
    dist_to_low_52w: float = np.nan
    wyckoff_phase: str = "Unknown"
    volume_accum_days: int = 0
    passed_filters: bool = False
    filter_details: dict[str, bool] = field(default_factory=dict)
    error: str = ""


# ======================================================================
# Checkpointing
# ======================================================================

_CHECKPOINT_PATH = OUTPUT_DIR / "_checkpoint.json"


def save_checkpoint(processed: set[str]) -> None:
    """Save the set of already-processed tickers."""
    if not ENABLE_CHECKPOINT:
        return
    try:
        data = {
            "processed": sorted(processed),
            "timestamp": datetime.now().isoformat(),
        }
        _CHECKPOINT_PATH.write_text(json.dumps(data))
    except Exception as exc:
        logger.warning("Failed to save checkpoint: %s", exc)


def load_checkpoint() -> set[str]:
    """Load previously-processed tickers from checkpoint."""
    if not _CHECKPOINT_PATH.exists():
        return set()
    try:
        data = json.loads(_CHECKPOINT_PATH.read_text())
        return set(data.get("processed", []))
    except Exception:
        return set()


def clear_checkpoint() -> None:
    """Remove the checkpoint file."""
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink(missing_ok=True)


# ======================================================================
# Single-ticker scan
# ======================================================================

def scan_single_from_df(
    ticker_info: TickerInfo,
    df: pd.DataFrame,
) -> ScanResult:
    """
    Run indicators → filters → score on an already-downloaded DataFrame.

    Returns a ScanResult regardless of whether the ticker passes or fails.
    """
    ticker = ticker_info.ticker

    try:
        if df is None or df.empty or len(df) < 20:
            return ScanResult(
                ticker=ticker,
                name=ticker_info.name,
                sector=ticker_info.sector,
                industry=ticker_info.industry,
                is_etf=ticker_info.is_etf,
                error="Insufficient data",
            )

        close = df["Close"].iloc[-1]

        # ---- 2. Indicators ----
        df = compute_all_indicators(df)

        # ---- 3. Filters ----
        filter_results = run_all_filters(df)
        passed = filter_results.all_passed()
        filter_map = {
            "min_price": filter_results.min_price.passed,
            "min_volume": filter_results.min_volume.passed,
            "bear_market": filter_results.bear_market.passed,
            "consolidation": filter_results.consolidation.passed,
            "volume_accumulation": filter_results.volume_accumulation.passed,
            "obv_divergence": filter_results.obv_divergence.passed,
            "cmf_positive": filter_results.cmf_positive.passed,
            "ad_slope": filter_results.ad_slope.passed,
            "volatility_contraction": filter_results.volatility_contraction.passed,
        }

        # ---- 4. Score ----
        sb = score_ticker(df)

        # ---- 5. Extract snapshot values ----
        obv_val = df["OBV"].iloc[-1] if "OBV" in df.columns else np.nan
        cmf_val = df["CMF"].iloc[-1] if "CMF" in df.columns else np.nan
        ad_val = df["AD"].iloc[-1] if "AD" in df.columns else np.nan
        atr14_val = df["ATR14"].iloc[-1] if "ATR14" in df.columns else np.nan
        rsi14_val = df["RSI14"].iloc[-1] if "RSI14" in df.columns else np.nan
        dist_low = df["DistToLow52W"].iloc[-1] if "DistToLow52W" in df.columns else np.nan
        phase = df["WyckoffPhase"].iloc[-1] if "WyckoffPhase" in df.columns else "Unknown"
        vol_accum_days = (
            int(df["_VolAccumDays"].iloc[-1]) if "_VolAccumDays" in df.columns else 0
        )

        return ScanResult(
            ticker=ticker,
            name=ticker_info.name,
            sector=ticker_info.sector,
            industry=ticker_info.industry,
            is_etf=ticker_info.is_etf,
            close=close,
            score=sb,
            obv=obv_val,
            cmf=cmf_val,
            ad=ad_val,
            atr14=atr14_val,
            rsi14=rsi14_val,
            dist_to_low_52w=dist_low,
            wyckoff_phase=phase,
            volume_accum_days=vol_accum_days,
            passed_filters=passed,
            filter_details=filter_map,
        )

    except Exception as exc:
        logger.debug("Error scanning %s: %s", ticker, exc)
        return ScanResult(
            ticker=ticker,
            name=ticker_info.name,
            is_etf=ticker_info.is_etf,
            error=str(exc),
        )


def scan_single(
    ticker_info: TickerInfo,
    force_download: bool = False,
) -> ScanResult:
    """
    Full scan pipeline for one ticker:
      download → indicators → filters → score.

    Prefer scan_single_from_df() when data is already downloaded.
    """
    ticker = ticker_info.ticker
    try:
        df = download_ticker(ticker, force=force_download)
        return scan_single_from_df(ticker_info, df)
    except Exception as exc:
        logger.debug("Error scanning %s: %s", ticker, exc)
        return ScanResult(
            ticker=ticker,
            name=ticker_info.name,
            is_etf=ticker_info.is_etf,
            error=str(exc),
        )


# ======================================================================
# Full scan orchestration
# ======================================================================

@dataclass
class ScanReport:
    """Aggregated results from a full scan run."""
    results: list[ScanResult] = field(default_factory=list)
    total_tickers: int = 0
    successful: int = 0
    failed: int = 0
    passed_filters: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_scan(
    stock_universe: list[TickerInfo] | None = None,
    etf_universe: list[TickerInfo] | None = None,
    force_download: bool = False,
    resume: bool = True,
) -> ScanReport:
    """
    Two-phase parallel scan across the entire ticker universe.

    Phase 1 — Download: parallel (ThreadPoolExecutor) data fetch via
    downloader.download_batch().  Tickers already cached are skipped
    unless force_download=True.  Bar the first ever run this phase is
    fast (most tickers only need a few incremental days).

    Phase 2 — Analyse: parallel indicator computation + scoring via
    ThreadPoolExecutor on the cached DataFrames.  This is CPU-bound
    and scales well with more workers.

    Args:
        stock_universe: Optional pre-built list of stocks.
        etf_universe: Optional pre-built list of ETFs.
        force_download: If True, re-download all data.
        resume: If True, load checkpoint and skip already-analysed tickers.

    Returns:
        ScanReport with ranked results.
    """

    start_time = time.time()

    # ---- Build universe ----
    if stock_universe is None and etf_universe is None:
        stock_universe, etf_universe = build_ticker_universe(
            include_stocks=True,
            include_etfs=True,
        )

    all_tickers: list[TickerInfo] = []
    if stock_universe:
        all_tickers.extend(stock_universe)
    if etf_universe:
        all_tickers.extend(etf_universe)

    # Deduplicate by ticker
    seen: set[str] = set()
    unique: list[TickerInfo] = []
    for ti in all_tickers:
        tup = ti.ticker.upper()
        if tup not in seen:
            seen.add(tup)
            unique.append(ti)
    all_tickers = unique

    # ---- Phase 1: Parallel download ----
    logger.info("Phase 1/2: downloading data for %d tickers (%d threads)...",
                len(all_tickers), DOWNLOAD_THREADS)

    # On the first ever run, everything is a full download.
    # On subsequent runs, download_batch() still calls download_ticker()
    # which does incremental fetch for already-cached symbols.
    download_batch(all_tickers, desc="Downloading", force=force_download)

    # ---- Phase 2: Parallel analyse ----
    # Load checkpoint to skip already-scored tickers
    processed_set = load_checkpoint() if resume else set()

    # Build the analyse queue — every ticker whose CSV exists on disk
    analyse_queue: list[TickerInfo] = []
    skipped_no_cache = 0
    for ti in all_tickers:
        if ti.ticker in processed_set:
            continue
        path = CACHE_DIR / f"{ti.ticker.replace('/', '_').replace('\\\\', '_')}.csv"
        if path.exists():
            analyse_queue.append(ti)
        else:
            skipped_no_cache += 1

    logger.info(
        "Phase 2/2: analysing %d tickers (%d threads) "
        "— %d already processed, %d no data on disk.",
        len(analyse_queue), SCAN_THREADS,
        len(processed_set), skipped_no_cache,
    )

    results: list[ScanResult] = []
    successful: int = 0
    failed: int = 0
    passed: int = 0

    # Also include previously-processed results from the last run's parquet
    prev_parquet = OUTPUT_DIR / f"Top{TOP_N_PARQUET}.parquet"
    prev_results: dict[str, ScanResult] = {}
    if resume and prev_parquet.exists():
        try:
            prev_df = pd.read_parquet(prev_parquet)
            for _, row in prev_df.iterrows():
                sr = ScanResult(
                    ticker=row.get("Ticker", ""),
                    name=row.get("Name", ""),
                    sector=row.get("Sector", ""),
                    industry=row.get("Industry", ""),
                    is_etf=bool(row.get("IsETF", False)),
                    close=float(row.get("Close", 0)),
                    score=ScoreBreakdown(
                        total=float(row.get("Score", 0)),
                        trend=float(row.get("TrendScore", 0)),
                        volume=float(row.get("VolumeScore", 0)),
                        accumulation=float(row.get("AccumulationScore", 0)),
                        volatility=float(row.get("CompressionScore", 0)),
                        structure=float(row.get("StructureScore", 0)),
                    ),
                    obv=row.get("OBV", np.nan),
                    cmf=row.get("CMF", np.nan),
                    ad=row.get("AD", np.nan),
                    atr14=row.get("ATR14", np.nan),
                    rsi14=row.get("RSI14", np.nan),
                    dist_to_low_52w=row.get("DistToLow52W", np.nan),
                    wyckoff_phase=str(row.get("WyckoffPhase", "Unknown")),
                    volume_accum_days=int(row.get("VolAccumDays", 0)),
                    passed_filters=bool(row.get("PassedFilters", False)),
                )
                prev_results[sr.ticker] = sr
        except Exception as exc:
            logger.debug("Could not load previous scan results: %s", exc)

    # Analyse in parallel
    with ThreadPoolExecutor(max_workers=SCAN_THREADS) as executor:
        futures = {}
        for ti in analyse_queue:
            # Load DF in main thread (I/O-bound single file read is fast)
            # then submit the CPU-heavy part
            futures[executor.submit(_analyse_one_ticker, ti)] = ti

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Analysing",
            unit="ticker",
        ):
            ti = futures[future]
            try:
                result = future.result(timeout=120)
            except Exception as exc:
                logger.warning("Analysis error for %s: %s", ti.ticker, exc)
                result = ScanResult(ticker=ti.ticker, error=str(exc))

            results.append(result)

            if result.error:
                failed += 1
            else:
                successful += 1
                if result.passed_filters:
                    passed += 1

            processed_set.add(ti.ticker)

            # Checkpoint every N tickers
            if len(processed_set) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(processed_set)

    # Merge previous results for tickers we didn't re-analyse
    for ticker, sr in prev_results.items():
        if ticker not in processed_set:
            results.append(sr)

    # Final checkpoint
    save_checkpoint(processed_set)

    # Sort by score descending
    results.sort(key=lambda r: r.score.total, reverse=True)

    elapsed = time.time() - start_time

    report = ScanReport(
        results=results,
        total_tickers=len(all_tickers),
        successful=successful + len(prev_results),
        failed=failed,
        passed_filters=passed,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "Scan complete: %d successful, %d failed, %d passed filters, %.1f seconds.",
        successful + len(prev_results), failed, passed, elapsed,
    )

    return report


def _analyse_one_ticker(ticker_info: TickerInfo) -> ScanResult:
    """Load cached CSV and run the analysis pipeline.  Sits inside a ThreadPool."""
    ticker = ticker_info.ticker
    try:
        df = _load_cache(ticker)
        if df is None or df.empty or len(df) < 20:
            return ScanResult(
                ticker=ticker,
                name=ticker_info.name,
                sector=ticker_info.sector,
                industry=ticker_info.industry,
                is_etf=ticker_info.is_etf,
                error="No cached data",
            )
        return scan_single_from_df(ticker_info, df)
    except Exception as exc:
        return ScanResult(ticker=ticker, name=ticker_info.name, error=str(exc))


# ======================================================================
# Parallel indicator computation (alternative fast path)
# ======================================================================

def run_parallel_indicator_scan(
    tickers: list[TickerInfo],
    max_workers: int = SCAN_THREADS,
) -> list[ScanResult]:
    """
    Compute indicators and scores in parallel using ThreadPoolExecutor.

    Data must already be cached. This is the fast path for re-scans.
    """
    results: list[ScanResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyse_one_ticker, ti): ti for ti in tickers
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Parallel scan",
            unit="ticker",
        ):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as exc:
                ti = futures[future]
                logger.warning("Scan timeout/error for %s: %s", ti.ticker, exc)
                results.append(ScanResult(
                    ticker=ti.ticker,
                    error=str(exc),
                ))

    results.sort(key=lambda r: r.score.total, reverse=True)
    return results
