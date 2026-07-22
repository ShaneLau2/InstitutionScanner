#!/usr/bin/env python3
"""
main.py — CLI entry point for the Institutional Accumulation Scanner.

Usage:
    python main.py scan                    # Full scan: stocks + ETFs (uses cache when available)
    python main.py scan --stocks-only      # Stocks only
    python main.py scan --etfs-only        # ETFs only
    python main.py scan --force-download   # Re-download all data
    python main.py scan --resume           # Resume from checkpoint (default)
    python main.py scan --no-resume        # Start fresh scan
    python main.py scan --tickers AAPL,TLT # Scan specific tickers only
    python main.py report                  # Re-generate report from cached data
    python main.py report --top 100        # Top 100 instead of 50
    python main.py download                # Download data only (no scan)
    python main.py download --stocks-only  # Stocks only
    python main.py clean                   # Clear all cached data and checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path so imports work from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CACHE_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    TOP_N_PARQUET,
    TOP_N_REPORT,
)
from downloader import (
    TickerInfo,
    build_ticker_universe,
    download_batch,
    download_ticker,
    prefilter_by_market_cap,
)
from scanner import (
    ScanReport,
    clear_checkpoint,
    run_scan,
    run_parallel_indicator_scan,
)
from report import (
    export_all,
    print_scan_summary,
    print_terminal_report,
)


# ======================================================================
# Logging setup
# ======================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with console and file handlers."""
    root = logging.getLogger("scanner_gui")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove old handlers to avoid duplicate output
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # File handler
    log_path = LOG_DIR / f"scan_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(fh)

    root.info("Logging to %s", log_path)


# ======================================================================
# CLI Commands
# ======================================================================

def cmd_scan(args: argparse.Namespace) -> int:
    """Run the full accumulation scan."""
    logger = logging.getLogger("scanner_gui")

    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only

    # Determine which markets to scan
    include_a_shares = not args.us_only
    include_us = args.us_only or args.include_us
    if args.a_shares_only:
        include_us = False
        include_stocks = True
        include_etfs = True

    # Determine market label for output filenames
    if args.us_only:
        market_label = "us"
    elif args.include_us:
        market_label = "all"
    else:
        market_label = "a_share"

    # ---- Cache check: skip re-download if today's results already exist ----
    if not args.tickers and not args.force_download:
        from datetime import date as _today_date
        today_str = _today_date.today().strftime("%Y%m%d")
        existing_csv = OUTPUT_DIR / f"{market_label}_{today_str}_AllResults.csv"
        existing_parquet = OUTPUT_DIR / f"{market_label}_{today_str}_AllResults.parquet"
        if existing_csv.exists() or existing_parquet.exists():
            logger.info("今日 %s 市场已分析完成（%s），跳过重新下载与分析。", market_label, existing_csv.name if existing_csv.exists() else existing_parquet.name)
            logger.info("如需强制重新扫描，请使用 --force-download 参数。")
            # Load results from parquet for terminal report
            if existing_parquet.exists():
                try:
                    import pandas as pd
                    from scanner import ScanResult, ScoreBreakdown
                    import numpy as np
                    df = pd.read_parquet(existing_parquet)
                    results: list[ScanResult] = []
                    for _, row in df.iterrows():
                        sr = ScanResult(
                            ticker=str(row.get("Ticker", "")),
                            market=str(row.get("Market", market_label)),
                            name=str(row.get("Name", "")),
                            sector=str(row.get("Sector", "")),
                            industry=str(row.get("Industry", "")),
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
                            style=str(row.get("Style", "均衡")),
                        )
                        results.append(sr)
                    print_terminal_report(results, n=args.top)
                    total = len(results)
                    passed = sum(1 for r in results if r.passed_filters)
                    from dataclasses import dataclass, field
                    from datetime import datetime
                    from scanner import ScanReport
                    report = ScanReport(
                        results=results,
                        total_tickers=total,
                        successful=total,
                        passed_filters=passed,
                    )
                    print_scan_summary(report)
                except Exception as exc:
                    logger.warning("加载已有结果失败，将重新扫描: %s", exc)
                else:
                    return 0

    # Build universe or use specific tickers
    if args.tickers:
        symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        # Detect market by suffix
        stock_universe: list[TickerInfo] = []
        etf_universe: list[TickerInfo] = []
        for s in symbols:
            mkt = "us" if not any(s.endswith(suffix) for suffix in (".SH", ".SZ", ".BJ")) else "a_share"
            ti = TickerInfo(ticker=s, market=mkt)
            stock_universe.append(ti)
        logger.info("Scanning %d specified tickers: %s", len(symbols), ", ".join(symbols))
    else:
        if not include_a_shares and include_us:
            logger.info("Building US stock universe only...")
            stock_universe, etf_universe = build_ticker_universe(
                include_stocks=False,
                include_etfs=False,
                include_us=True,
            )
        elif include_stocks or include_etfs:
            logger.info("Building ticker universe (stocks=%s, ETFs=%s, US=%s)...",
                        include_stocks, include_etfs, include_us)
            stock_universe, etf_universe = build_ticker_universe(
                include_stocks=include_stocks,
                include_etfs=include_etfs,
                include_us=include_us,
            )
        else:
            stock_universe, etf_universe = [], []
        logger.info(
            "Universe: %d stocks, %d ETFs — %d total.",
            len(stock_universe), len(etf_universe),
            len(stock_universe) + len(etf_universe),
        )

        # Pre-filter: skip tickers whose known market cap is below threshold
        stock_universe = prefilter_by_market_cap(stock_universe, market_label)

    # Run the scan
    report = run_scan(
        stock_universe=stock_universe,
        etf_universe=etf_universe,
        force_download=args.force_download,
        resume=not args.no_resume,
        data_source=args.data_source,
        include_us=include_us,
        market=market_label,
    )

    if report.successful == 0:
        logger.error("没有可用行情数据，扫描失败；请检查网络或数据源后重试。")
        print_scan_summary(report)
        return 2

    # Export results
    csv_path, parquet_path, full_csv, full_parquet = export_all(
        report.results,
        top_n_csv=args.top,
        top_n_parquet=args.top_parquet,
        market=market_label,
    )

    # Terminal report
    print_terminal_report(report.results, n=args.top)
    print_scan_summary(report)

    logger.info("Top CSV:    %s", csv_path)
    logger.info("Top PQ:     %s", parquet_path)
    logger.info("All CSV:    %s", full_csv)
    logger.info("All PQ:     %s", full_parquet)

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """
    Re-generate reports from already-cached data.
    Useful for re-scoring without re-downloading.
    """
    logger = logging.getLogger("scanner_gui")

    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only

    stock_universe, etf_universe = build_ticker_universe(
        include_stocks=include_stocks,
        include_etfs=include_etfs,
    )

    all_tickers = list(stock_universe) + list(etf_universe)

    logger.info("Re-scanning %d cached tickers...", len(all_tickers))
    results = run_parallel_indicator_scan(all_tickers)

    csv_path, parquet_path, full_csv, full_parquet = export_all(results, top_n_csv=args.top, top_n_parquet=args.top_parquet)
    print_terminal_report(results, n=args.top)

    logger.info("Top CSV:    %s", csv_path)
    logger.info("Top PQ:     %s", parquet_path)
    logger.info("All CSV:    %s", full_csv)
    logger.info("All PQ:     %s", full_parquet)

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download data only — no scan, no report."""
    logger = logging.getLogger("scanner_gui")

    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only

    if args.tickers:
        symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        all_tickers = [TickerInfo(ticker=s) for s in symbols]
    else:
        stock_universe, etf_universe = build_ticker_universe(
            include_stocks=include_stocks,
            include_etfs=include_etfs,
            include_us=getattr(args, "include_us", False),
        )
        all_tickers = list(stock_universe) + list(etf_universe)

    logger.info("Downloading data for %d tickers...", len(all_tickers))
    results = download_batch(all_tickers, desc="Downloading")
    logger.info("Successfully downloaded %d tickers.", len(results))

    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove all cached data and checkpoints."""
    logger = logging.getLogger("scanner_gui")
    import shutil

    if args.cache_only:
        dirs = [CACHE_DIR]
    elif args.output_only:
        dirs = [OUTPUT_DIR]
    else:
        dirs = [CACHE_DIR, OUTPUT_DIR]

    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared: %s", d)

    # Recreate cache subdirectories if cache was cleaned
    if CACHE_DIR in dirs or not args.output_only:
        for sub in ("a_share", "us"):
            (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)

    clear_checkpoint()
    logger.info("Checkpoint cleared.")
    return 0


# ======================================================================
# Argument parser
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ScannerGui",
        description="Institutional Accumulation Scanner — find A-share stocks & ETFs "
                    "being quietly accumulated by institutions during bear markets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- scan ----
    scan_p = sub.add_parser("scan", help="Run the full accumulation scan")
    scan_p.add_argument("--a-shares-only", action="store_true", help="Scan only A-shares (default includes both)")
    scan_p.add_argument("--us-only", action="store_true", help="Scan only US stocks and ETFs")
    scan_p.add_argument("--include-us", action="store_true", help="Include US stocks alongside A-shares")
    scan_p.add_argument("--stocks-only", action="store_true", help="Scan only stocks")
    scan_p.add_argument("--etfs-only", action="store_true", help="Scan only ETFs")
    scan_p.add_argument("--force-download", action="store_true",
                        help="Re-download all data (ignore cache)")
    scan_p.add_argument("--no-resume", action="store_true",
                        help="Do not resume from checkpoint — start fresh")
    scan_p.add_argument("--cache-first", action="store_true",
                        help="Prefer cached data and skip re-downloading unchanged tickers")
    scan_p.add_argument("--data-source", choices=("eastmoney", "sina", "tencent"), default="eastmoney", help="A股行情数据源（美股自动使用yfinance）")
    scan_p.add_argument("--top", type=int, default=TOP_N_REPORT,
                        help=f"Number of tickers in the terminal report (default: {TOP_N_REPORT})")
    scan_p.add_argument("--top-parquet", type=int, default=TOP_N_PARQUET,
                        help=f"Number of tickers in the Parquet file (default: {TOP_N_PARQUET})")
    scan_p.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated list of specific tickers to scan")
    scan_p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # ---- report ----
    report_p = sub.add_parser("report", help="Re-generate report from cached data")
    report_p.add_argument("--stocks-only", action="store_true")
    report_p.add_argument("--etfs-only", action="store_true")
    report_p.add_argument("--top", type=int, default=TOP_N_REPORT)
    report_p.add_argument("--top-parquet", type=int, default=TOP_N_PARQUET)
    report_p.add_argument("--verbose", "-v", action="store_true")

    # ---- download ----
    dl_p = sub.add_parser("download", help="Download data only (no scan)")
    dl_p.add_argument("--stocks-only", action="store_true")
    dl_p.add_argument("--etfs-only", action="store_true")
    dl_p.add_argument("--tickers", type=str, default=None)
    dl_p.add_argument("--verbose", "-v", action="store_true")

    # ---- clean ----
    clean_p = sub.add_parser("clean", help="Clear cached data and outputs")
    clean_p.add_argument("--cache-only", action="store_true", help="Clear only cache")
    clean_p.add_argument("--output-only", action="store_true", help="Clear only outputs")
    clean_p.add_argument("--verbose", "-v", action="store_true")

    return parser


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(verbose=getattr(args, "verbose", False))

    commands = {
        "scan": cmd_scan,
        "report": cmd_report,
        "download": cmd_download,
        "clean": cmd_clean,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("scanner_gui").exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
