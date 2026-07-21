#!/usr/bin/env python3
"""
main.py — CLI entry point for the Institutional Accumulation Scanner.

Usage:
    python main.py scan                    # Full scan: stocks + ETFs
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
    root = logging.getLogger("institution_scanner")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

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
    logger = logging.getLogger("institution_scanner")

    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only

    # Build universe or use specific tickers
    if args.tickers:
        symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        stock_universe = [TickerInfo(ticker=s) for s in symbols]
        etf_universe: list[TickerInfo] = []
        logger.info("Scanning %d specified tickers: %s", len(symbols), ", ".join(symbols))
    else:
        logger.info("Building ticker universe (stocks=%s, ETFs=%s)...", include_stocks, include_etfs)
        stock_universe, etf_universe = build_ticker_universe(
            include_stocks=include_stocks,
            include_etfs=include_etfs,
        )
        logger.info(
            "Universe: %d stocks, %d ETFs — %d total.",
            len(stock_universe), len(etf_universe),
            len(stock_universe) + len(etf_universe),
        )

    # Run the scan
    report = run_scan(
        stock_universe=stock_universe,
        etf_universe=etf_universe,
        force_download=args.force_download,
        resume=not args.no_resume,
    )

    # Export results
    csv_path, parquet_path, full_csv, full_parquet = export_all(
        report.results,
        top_n_csv=args.top,
        top_n_parquet=args.top_parquet,
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
    logger = logging.getLogger("institution_scanner")

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
    logger = logging.getLogger("institution_scanner")

    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only

    if args.tickers:
        symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        all_tickers = [TickerInfo(ticker=s) for s in symbols]
    else:
        stock_universe, etf_universe = build_ticker_universe(
            include_stocks=include_stocks,
            include_etfs=include_etfs,
        )
        all_tickers = list(stock_universe) + list(etf_universe)

    logger.info("Downloading data for %d tickers...", len(all_tickers))
    results = download_batch(all_tickers, desc="Downloading")
    logger.info("Successfully downloaded %d tickers.", len(results))

    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove all cached data and checkpoints."""
    logger = logging.getLogger("institution_scanner")
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

    clear_checkpoint()
    logger.info("Checkpoint cleared.")
    return 0


# ======================================================================
# Argument parser
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="InstitutionScanner",
        description="Institutional Accumulation Scanner — find stocks & ETFs "
                    "being quietly accumulated by institutions during bear markets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- scan ----
    scan_p = sub.add_parser("scan", help="Run the full accumulation scan")
    scan_p.add_argument("--stocks-only", action="store_true", help="Scan only stocks")
    scan_p.add_argument("--etfs-only", action="store_true", help="Scan only ETFs")
    scan_p.add_argument("--force-download", action="store_true",
                        help="Re-download all data (ignore cache)")
    scan_p.add_argument("--no-resume", action="store_true",
                        help="Do not resume from checkpoint — start fresh")
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
        logging.getLogger("institution_scanner").exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
