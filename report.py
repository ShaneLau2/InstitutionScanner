"""
report.py — Output generation: CSV, Parquet, and terminal reports.

Produces:
  - Top50.csv:          the 50 highest-scoring tickers with key metrics.
  - AllResults.csv:     every scored ticker, sorted by score.
  - Top200.parquet:     the top 200 in Parquet format.
  - AllResults.parquet: every scored ticker in Parquet format.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import OUTPUT_DIR, TOP_N_PARQUET, TOP_N_REPORT
from scanner import ScanResult, ScanReport

logger = logging.getLogger("scanner_gui.report")


# ======================================================================
# Filename helpers
# ======================================================================


def _today_str() -> str:
    """Return today's date as YYYYMMDD string."""
    return date.today().strftime("%Y%m%d")


def market_output_filename(base: str, market: str = "a_share") -> str:
    """
    Build a market+date prefixed filename.

    E.g. market_output_filename("Top50.csv", "us") -> "us_20260722_Top50.csv"
    """
    return f"{market}_{_today_str()}_{base}"


def resolve_latest_file(base: str, market: str = "a_share") -> Path:
    """
    Return the path to today's market-specific file, falling back to
    the plain (legacy) filename.
    """
    prefixed = OUTPUT_DIR / market_output_filename(base, market)
    if prefixed.exists():
        return prefixed
    legacy = OUTPUT_DIR / base
    if legacy.exists():
        return legacy
    return prefixed  # caller handles non-existence


# ======================================================================
# Data export helpers
# ======================================================================

def _quality_label(result: ScanResult) -> str:
    signal_count = int(result.filter_details.get("signal_count", 0))
    score = float(result.score.total)
    if result.passed_filters and (
        (signal_count >= 5 and score >= 40) or (signal_count >= 4 and score >= 48)
    ):
        return "强候选"
    if result.passed_filters and signal_count >= 4:
        return "候选"
    if score >= 35 or signal_count >= 3:
        return "观察"
    return "普通"


def _rankable_results(results: list[ScanResult]) -> list[ScanResult]:
    valid = [r for r in results if not r.error]
    passed = [r for r in valid if r.passed_filters]
    candidates = passed if passed else valid
    return sorted(
        candidates,
        key=lambda r: (
            float(r.score.total),
            int(r.passed_filters),
            int(r.filter_details.get("signal_count", 0)),
        ),
        reverse=True,
    )


def _results_to_dataframe(results: list[ScanResult]) -> pd.DataFrame:
    """Convert ScanResult list to a sorted, clean DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Ticker": r.ticker,
            "Name": r.name,
            "MarketCap": r.market_cap_val,
            "Turnover": round(r.turnover, 2) if r.turnover else 0.0,
            "Market": r.market,
            "Sector": r.sector,
            "Industry": r.industry,
            "IsETF": r.is_etf,
            "AssetType": r.asset_type,
            "Style": r.style,
            "Quality": _quality_label(r),
            "Close": r.close,
            "Score": round(r.score.total, 2),
            "TrendScore": round(r.score.trend, 2),
            "VolumeScore": round(r.score.volume, 2),
            "AccumulationScore": round(r.score.accumulation, 2),
            "CompressionScore": round(r.score.volatility, 2),
            "StructureScore": round(r.score.structure, 2),
            "OBV": r.obv if not np.isnan(r.obv) else None,
            "CMF": round(r.cmf, 4) if not np.isnan(r.cmf) else None,
            "AD": r.ad if not np.isnan(r.ad) else None,
            "ATR14": round(r.atr14, 4) if not np.isnan(r.atr14) else None,
            "RSI14": round(r.rsi14, 2) if not np.isnan(r.rsi14) else None,
            "DistToLow52W": round(r.dist_to_low_52w, 2) if not np.isnan(r.dist_to_low_52w) else None,
            "WyckoffPhase": r.wyckoff_phase,
            "VolAccumDays": r.volume_accum_days,
            "SignalCount": r.filter_details.get("signal_count", 0),
            "FilterCount": r.filter_details.get("filter_count", 0),
            "PassedFilters": r.passed_filters,
            "OBV_Div": r.filter_details.get("obv_divergence", False),
            "CMF_Pos": r.filter_details.get("cmf_positive", False),
            "AD_SlopePos": r.filter_details.get("ad_slope", False),
            "BearMarket": r.filter_details.get("bear_market", False),
            "Consolidation": r.filter_details.get("consolidation", False),
            "VolAccum": r.filter_details.get("volume_accumulation", False),
            "VolContract": r.filter_details.get("volatility_contraction", False),
            "Error": r.error if r.error else "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["PassedFilters", "Score", "SignalCount"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


# ======================================================================
# CSV Export
# ======================================================================

def export_top_csv(results: list[ScanResult], n: int = TOP_N_REPORT, market: str = "a_share") -> Path:
    """
    Export the top *n* tickers to {market}_{date}_Top{n}.csv.

    Returns the path to the generated file.
    """
    df = _results_to_dataframe(_rankable_results(results))
    top = df.head(n)

    path = OUTPUT_DIR / market_output_filename(f"Top{n}.csv", market)
    top.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Exported Top %d to %s", n, path)
    return path


def export_full_csv(results: list[ScanResult], market: str = "a_share") -> Path:
    """
    Export ALL scored tickers to {market}_{date}_AllResults.csv.
    """
    df = _results_to_dataframe(results)
    path = OUTPUT_DIR / market_output_filename("AllResults.csv", market)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Exported all %d results to %s", len(df), path)
    return path


# ======================================================================
# Parquet Export
# ======================================================================

def export_top_parquet(results: list[ScanResult], n: int = TOP_N_PARQUET, market: str = "a_share") -> Path:
    """
    Export top *n* tickers to {market}_{date}_Top{n}.parquet.

    Returns the path to the generated file.
    """
    df = _results_to_dataframe(_rankable_results(results))
    top = df.head(n)

    path = OUTPUT_DIR / market_output_filename(f"Top{n}.parquet", market)
    table = pa.Table.from_pandas(top)
    pq.write_table(table, path)
    logger.info("Exported Top %d to %s", n, path)
    return path


# ======================================================================
# Full export
# ======================================================================

def export_all(
    results: list[ScanResult],
    top_n_csv: int = TOP_N_REPORT,
    top_n_parquet: int = TOP_N_PARQUET,
    market: str = "a_share",
) -> tuple[Path, Path, Path, Path]:
    """Export CSV, Parquet, and full results. Returns (csv_path, parquet_path, full_csv, full_parquet)."""
    csv_path = export_top_csv(results, n=top_n_csv, market=market)
    parquet_path = export_top_parquet(results, n=top_n_parquet, market=market)
    full_csv = export_full_csv(results, market=market)
    full_parquet_path = export_full_parquet(results, market=market)
    return csv_path, parquet_path, full_csv, full_parquet_path


def export_full_parquet(results: list[ScanResult], market: str = "a_share") -> Path:
    """
    Export ALL scored tickers to {market}_{date}_AllResults.parquet.
    """
    df = _results_to_dataframe(results)
    path = OUTPUT_DIR / market_output_filename("AllResults.parquet", market)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    logger.info("Exported all %d results to %s", len(df), path)
    return path


# ======================================================================
# Terminal Report
# ======================================================================

def _build_reasons(result: ScanResult) -> list[str]:
    """Build a list of human-readable reasons why this ticker scored well."""
    reasons: list[str] = []

    if result.filter_details.get("bear_market"):
        reasons.append("✓ MA200 declining, long-term bear market")

    if result.filter_details.get("volume_accumulation"):
        reasons.append(f"✓ Sustained volume accumulation ({result.volume_accum_days} days)")

    if result.filter_details.get("obv_divergence"):
        reasons.append("✓ OBV Bullish Divergence detected")

    if result.filter_details.get("cmf_positive"):
        cmf_str = f"{result.cmf:.3f}" if not np.isnan(result.cmf) else "N/A"
        reasons.append(f"✓ CMF Positive ({cmf_str})")

    if result.filter_details.get("ad_slope"):
        reasons.append("✓ A/D Line rising")

    if result.filter_details.get("volatility_contraction"):
        reasons.append("✓ Volatility contraction (ATR/BB)")

    if result.filter_details.get("consolidation"):
        reasons.append("✓ Bottom consolidation pattern")

    if result.wyckoff_phase not in ("Unknown", ""):
        reasons.append(f"✓ Wyckoff: {result.wyckoff_phase}")

    return reasons


def print_terminal_report(results: list[ScanResult], n: int = TOP_N_REPORT) -> None:
    """
    Print a formatted Top-N report to stdout.

    Each entry shows rank, ticker, score, and specific reasons.
    """
    top = _rankable_results(results)[:n]
    if not top:
        print("\nNo tickers passed the accumulation filters.\n")
        return

    print()
    print("=" * 70)
    print(f"  INSTITUTIONAL ACCUMULATION SCANNER — TOP {min(n, len(top))}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()

    for i, result in enumerate(top, start=1):
        name_str = f" — {result.name}" if result.name else ""
        etf_tag = " [ETF]" if result.is_etf else ""
        sector_str = f" | {result.sector}" if result.sector else ""

        print(f"  {i:3d}. {result.ticker:<8s} "
              f"Score: {result.score.total:5.1f}{etf_tag}")
        if name_str.strip():
            print(f"      {name_str.strip()}{sector_str}")
        print(f"      Close: ¥{result.close:.2f} | "
              f"RSI14: {result.rsi14:.1f} | "
              f"ATR14: {result.atr14:.2f} | "
              f"Phase: {result.wyckoff_phase}")

        reasons = _build_reasons(result)
        for reason in reasons:
            print(f"      {reason}")

        print(f"      {'-' * 60}")


def print_scan_summary(report: ScanReport) -> None:
    """Print a one-paragraph scan summary to stdout."""
    print()
    print(f"Scan complete in {report.elapsed_seconds:.1f} seconds.")
    print(f"  Total tickers:    {report.total_tickers}")
    print(f"  Scanned:          {report.successful + report.failed}")
    print(f"  Successful:       {report.successful}")
    print(f"  Failed/No data:   {report.failed}")
    print(f"  Passed filters:   {report.passed_filters}")
    print()
