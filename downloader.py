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
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

from config import (
    CACHE_DIR,
    DOWNLOAD_RATE_LIMIT_PAUSE,
    DOWNLOAD_RETRIES,
    DOWNLOAD_THREADS,
    DOWNLOAD_TIMEOUT,
    HISTORY_YEARS,
    LOG_DIR,
    A_SHARE_CONFIG,
    US_STOCK_CONFIG,
    MARKET_CONFIGS,
    MAX_DOWNLOAD_ERRORS,
    MIN_MARKET_CAP,
    EXCLUDED_SECURITY_KEYWORDS,
    MIN_PRICE,
    MIN_VOLUME,
)

logger = logging.getLogger("scanner_gui.downloader")
logger.setLevel(logging.DEBUG)

# Attach a rotating file handler so we don't lose logs
_fh = logging.FileHandler(LOG_DIR / "downloader.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

_HTTP = requests.Session()
_HTTP.trust_env = True
_EASTMONEY_HOSTS = ("push2delay.eastmoney.com", "push2.eastmoney.com")
_EASTMONEY_HISTORY_HOSTS = ("push2delay.eastmoney.com", "push2his.eastmoney.com")
_UNIVERSE_CACHE_PATH = CACHE_DIR / "_a_share_universe.json"


def _eastmoney_get(path: str, params: dict[str, Any], history: bool = False) -> requests.Response:
    hosts = _EASTMONEY_HISTORY_HOSTS if history else _EASTMONEY_HOSTS
    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    for attempt in range(DOWNLOAD_RETRIES + 1):
        for host in hosts:
            try:
                response = _HTTP.get(
                    f"https://{host}{path}",
                    params=params,
                    headers=headers,
                    timeout=DOWNLOAD_TIMEOUT,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"东方财富接口连接失败: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TickerInfo:
    """Metadata for a single ticker, including which market it belongs to."""
    ticker: str
    market: str = "a_share"
    name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    is_etf: bool = False
    asset_type: str = "stock"
    market_cap: float | None = None


def _is_excluded_security_name(name: str) -> bool:
    normalized = re.sub(r"\s+", "", str(name or "")).upper()
    return any(keyword.upper() in normalized for keyword in EXCLUDED_SECURITY_KEYWORDS)


# ---------------------------------------------------------------------------
# Ticker universe builders
# ---------------------------------------------------------------------------

# A-share universe: a curated, free list of major Chinese stocks and ETFs.
# This preserves the scanner's logic while switching the market focus to A-shares.
_STATIC_A_STOCKS: list[tuple[str, str, str, str]] = [
    ("000001.SZ", "平安银行", "金融", "银行"),
    ("000002.SZ", "万科A", "地产", "房地产"),
    ("000333.SZ", "美的集团", "消费", "家电"),
    ("000538.SZ", "云南白药", "医药", "中药"),
    ("000858.SZ", "五粮液", "消费", "白酒"),
    ("002594.SZ", "比亚迪", "消费", "汽车"),
    ("002352.SZ", "顺丰控股", "物流", "物流"),
    ("300750.SZ", "宁德时代", "消费", "新能源"),
    ("300014.SZ", "亿纬锂能", "消费", "新能源"),
    ("300059.SZ", "东方财富", "金融", "互联网金融"),
    ("600000.SH", "浦发银行", "金融", "银行"),
    ("600036.SH", "招商银行", "金融", "银行"),
    ("600519.SH", "贵州茅台", "消费", "白酒"),
    ("601318.SH", "中国平安", "金融", "保险"),
    ("601166.SH", "兴业银行", "金融", "银行"),
    ("601857.SH", "中国石油", "能源", "石油"),
    ("601988.SH", "中国银行", "金融", "银行"),
    ("603259.SH", "药明康德", "医药", "生物医药"),
    ("603501.SH", "韦尔股份", "消费", "电子"),
    ("688981.SH", "中芯国际", "消费", "半导体"),
    ("688599.SH", "天合光能", "消费", "光伏"),
    ("600104.SH", "上汽集团", "消费", "汽车"),
    ("600028.SH", "中国石化", "能源", "石化"),
    ("600900.SH", "长江电力", "公用事业", "电力"),
    ("601899.SH", "紫金矿业", "采矿", "有色金属"),
    ("601989.SH", "中国重工", "工业", "装备"),
    ("603799.SH", "华友钴业", "消费", "新能源"),
]

_STATIC_A_ETFS: list[tuple[str, str]] = [
    ("510300.SH", "沪深300ETF"),
    ("510500.SH", "中证500ETF"),
    ("159901.SZ", "深证100ETF"),
    ("159915.SZ", "创业板ETF"),
    ("515000.SH", "上证红利ETF"),
    ("512690.SH", "科创50ETF"),
    ("159952.SZ", "医药ETF"),
    ("518880.SH", "黄金ETF"),
    ("159996.SZ", "新能源车ETF"),
    ("512980.SH", "证券ETF"),
    ("510880.SH", "红利ETF"),
    ("159915.SZ", "创业板ETF"),
    ("159997.SZ", "芯片ETF"),
    ("159996.SZ", "新能源ETF"),
]

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


def _is_rejected_stock_name(name: str) -> bool:
    normalized = str(name or "").upper().replace(" ", "")
    return "ST" in normalized or "退" in normalized or "退市" in normalized


def _fetch_a_share_stocks() -> list[TickerInfo]:
    """Fetch the complete Shanghai, Shenzhen and Beijing A-share universe."""
    params = {
        "pn": 1,
        "pz": 100,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f13,f14,f20,f100,f102",
    }
    try:
        response = _eastmoney_get("/api/qt/clist/get", params)
        data = response.json().get("data") or {}
        rows = list(data.get("diff") or [])
        if not rows:
            raise RuntimeError("东方财富未返回A股证券列表")
        total = int(data.get("total") or len(rows))
        for page in range(2, math.ceil(total / params["pz"]) + 1):
            params["pn"] = page
            response = _eastmoney_get("/api/qt/clist/get", params)
            rows.extend(((response.json().get("data") or {}).get("diff") or []))
        _UNIVERSE_CACHE_PATH.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception:
        if not _UNIVERSE_CACHE_PATH.exists():
            raise
        rows = json.loads(_UNIVERSE_CACHE_PATH.read_text(encoding="utf-8"))
        logger.warning("证券池接口不可用，使用本地缓存的 %d 只A股。", len(rows))

    tickers: list[TickerInfo] = []
    for row in rows:
        code = str(row.get("f12") or "").zfill(6)
        market = int(row.get("f13") or 0)
        if not code.isdigit() or len(code) != 6:
            continue
        name = str(row.get("f14") or "")
        if _is_rejected_stock_name(name):
            continue
        suffix = "SH" if market == 1 else "BJ" if code.startswith(("4", "8", "92")) else "SZ"
        market_cap = row.get("f20")
        tickers.append(TickerInfo(
            ticker=f"{code}.{suffix}",
            name=name,
            exchange={"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}[suffix],
            sector=str(row.get("f100") or ""),
            industry=str(row.get("f102") or ""),
            asset_type="stock",
            market_cap=float(market_cap) if isinstance(market_cap, (int, float)) and market_cap > 0 else None,
        ))
    if len(tickers) < 4000:
        raise RuntimeError(f"A股证券列表数量异常，仅获取到 {len(tickers)} 只")
    logger.info("Loaded %d A-share stocks from Eastmoney.", len(tickers))
    return tickers


def _fetch_a_share_etfs() -> list[TickerInfo]:
    """Fetch listed Shanghai and Shenzhen ETFs from Eastmoney."""
    params = {
        "pn": 1,
        "pz": 100,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:9,m:1+t:9",
        "fields": "f12,f13,f14,f20,f100,f102",
    }
    try:
        response = _eastmoney_get("/api/qt/clist/get", params)
        data = response.json().get("data") or {}
        rows = list(data.get("diff") or [])
        if not rows:
            raise RuntimeError("东方财富未返回ETF列表")
        total = int(data.get("total") or len(rows))
        for page in range(2, math.ceil(total / params["pz"]) + 1):
            params["pn"] = page
            response = _eastmoney_get("/api/qt/clist/get", params)
            rows.extend(((response.json().get("data") or {}).get("diff") or []))
    except Exception:
        logger.exception("获取全量ETF失败")
        return [
            TickerInfo(ticker=symbol, name=name, exchange="SSE/SZSE", is_etf=True, asset_type="etf")
            for symbol, name in _STATIC_A_ETFS
            if not _is_excluded_security_name(name)
        ]

    etfs: list[TickerInfo] = []
    allowed_prefixes = ("15", "16", "50", "51", "56", "58")
    for row in rows:
        code = str(row.get("f12") or "").zfill(6)
        market = int(row.get("f13") or 0)
        name = str(row.get("f14") or "")
        if not code.isdigit() or len(code) != 6 or not code.startswith(allowed_prefixes):
            continue
        if _is_excluded_security_name(name):
            continue
        if name.endswith(("R", "A")) or "分级" in name or "退市" in name:
            continue
        suffix = "SH" if market == 1 else "SZ"
        etfs.append(TickerInfo(
            ticker=f"{code}.{suffix}",
            name=name,
            exchange={"SH": "SSE", "SZ": "SZSE"}[suffix],
            is_etf=True,
            asset_type="etf",
            market_cap=float(row["f20"]) if isinstance(row.get("f20"), (int, float)) and row["f20"] > 0 else None,
        ))
    unique = {item.ticker: item for item in etfs}
    result = sorted(unique.values(), key=lambda item: item.ticker)
    logger.info("Loaded %d A-share ETFs from Eastmoney.", len(result))
    return result


def build_ticker_universe(
    include_stocks: bool = True,
    include_etfs: bool = True,
    include_us: bool = False,
) -> tuple[list[TickerInfo], list[TickerInfo]]:
    """
    Build the complete ticker universe.

    Args:
        include_stocks: Include A-share stocks.
        include_etfs: Include A-share ETFs.
        include_us: Also include US stocks and ETFs.

    Returns:
        (stocks, etfs) — two lists of TickerInfo.
        Each ticker is deduplicated by symbol.
    """
    stocks: dict[str, TickerInfo] = {}
    etfs: dict[str, TickerInfo] = {}

    if include_stocks:
        for ti in _fetch_a_share_stocks():
            if not ti.is_etf:
                key = ti.ticker.upper()
                if key not in stocks:
                    stocks[key] = ti

    if include_etfs:
        for ti in _fetch_a_share_etfs():
            key = ti.ticker.upper()
            if key not in etfs:
                etfs[key] = ti

    if include_us and _YFINANCE_AVAILABLE:
        logger.info("Fetching US stock universe (S&P 500 + NASDAQ 100)...")
        for ti in _fetch_us_stocks():
            if not ti.is_etf:
                key = ti.ticker.upper()
                if key not in stocks:
                    stocks[key] = ti
        logger.info("Adding US ETFs...")
        for ti in _fetch_us_etfs():
            key = ti.ticker.upper()
            if key not in etfs:
                etfs[key] = ti
    elif include_us:
        logger.warning("US stocks requested but yfinance not installed — skipping.")

    stock_list = sorted(stocks.values(), key=lambda x: (x.market, x.ticker))
    etf_list = sorted(etfs.values(), key=lambda x: (x.market, x.ticker))

    us_stocks = sum(1 for s in stock_list if s.market == "us")
    us_etfs = sum(1 for e in etf_list if e.market == "us")
    logger.info(
        "Universe built: %d stocks (%d US) + %d ETFs (%d US)",
        len(stock_list), us_stocks,
        len(etf_list), us_etfs,
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
        code, suffix = ticker.upper().split(".", 1)
        market = "1" if suffix == "SH" else "0"
        response = _eastmoney_get(
            "/api/qt/stock/get",
            {"secid": f"{market}.{code}", "fields": "f20"},
        )
        response.raise_for_status()
        mc = (response.json().get("data") or {}).get("f20")
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


def _download_from_sina(ticker: str) -> pd.DataFrame | None:
    code, suffix = ticker.upper().split(".", 1)
    if suffix == "BJ":
        return None
    symbol = ("sh" if suffix == "SH" else "sz") + code
    response = _HTTP.get(
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData",
        params={"symbol": symbol, "scale": 240, "ma": "no", "datalen": 1023},
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        timeout=DOWNLOAD_TIMEOUT,
    )
    response.raise_for_status()
    text = response.text
    match = re.search(r"var _data=\((\[.*?\])\);", text, re.S)
    if not match:
        return None
    rows = json.loads(match.group(1))
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={"day": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()


def _download_from_tencent(ticker: str) -> pd.DataFrame | None:
    code, suffix = ticker.upper().split(".", 1)
    if suffix == "BJ":
        return None
    prefix = "sh" if suffix == "SH" else "sz"
    symbol = f"{prefix}{code}"
    end_date = datetime.now()
    start_limit = end_date - timedelta(days=HISTORY_YEARS * 365 + 30)
    rows: list[list[str]] = []
    while end_date > start_limit:
        response = _HTTP.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={
                "param": f"{symbol},day,{start_limit:%Y-%m-%d},{end_date:%Y-%m-%d},640,qfq",
            },
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
        data = (response.json().get("data") or {}).get(symbol) or {}
        batch = data.get("qfqday") or data.get("day") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = pd.Timestamp(batch[0][0]).to_pydatetime()
        if oldest <= start_limit or oldest >= end_date:
            break
        end_date = oldest - timedelta(days=1)
        if len(batch) < 640:
            break
    if not rows:
        return None
    records = [row[:6] for row in rows]
    df = pd.DataFrame(records, columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    return df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])


def _download_from_eastmoney(ticker: str) -> pd.DataFrame | None:
    """
    Download full history for *ticker* from Eastmoney.
    Returns a DataFrame or None on failure.
    """
    attempts = DOWNLOAD_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=HISTORY_YEARS * 365 + 30)
            code, suffix = ticker.upper().split(".", 1)
            market = "1" if suffix == "SH" else "0"
            response = _eastmoney_get(
                "/api/qt/stock/kline/get",
                {
                    "secid": f"{market}.{code}",
                    "klt": 101,
                    "fqt": 1,
                    "beg": start_date.strftime("%Y%m%d"),
                    "end": end_date.strftime("%Y%m%d"),
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56",
                },
                history=True,
            )
            klines = ((response.json().get("data") or {}).get("klines") or [])
            if not klines:
                for fallback_loader in (_download_from_sina, _download_from_tencent):
                    try:
                        fallback = fallback_loader(ticker)
                        if fallback is not None and not fallback.empty:
                            return fallback
                    except Exception as fallback_exc:
                        logger.debug("Fallback failed for %s: %s", ticker, fallback_exc)
                logger.debug("Eastmoney returned no K-line data for %s", ticker)
                return None
            records = [line.split(",")[:6] for line in klines]
            df = pd.DataFrame(records, columns=["Date", "Open", "Close", "High", "Low", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for column in ("Open", "High", "Low", "Close", "Volume"):
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna(subset=["Close"])
            if df.empty:
                return None
            return df
        except Exception as exc:
            for fallback_loader in (_download_from_sina, _download_from_tencent):
                try:
                    fallback = fallback_loader(ticker)
                    if fallback is not None and not fallback.empty:
                        return fallback
                except Exception as fallback_exc:
                    logger.debug("Fallback failed for %s: %s", ticker, fallback_exc)
            msg = str(exc).lower()
            # 404 / delisted / timeout / curl errors — skip instantly
            if any(kw in msg for kw in ("404", "not found", "delisted", "no timezone", "timeout", "timed out", "no data found", "failed to perform", "curl")):
                return None
            # 401 / 429 rate limits — back off harder
            if "401" in msg or "429" in msg or "rate limit" in msg:
                delay = 5 + (attempt * 5)
                logger.debug(
                    "Rate-limited on %s (attempt %d/%d), backing off %ds...",
                    ticker, attempt, attempts, delay,
                )
                time.sleep(delay)
                continue
            logger.debug("Attempt %d/%d failed for %s: %s", attempt, attempts, ticker, exc)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return None


_DATA_SOURCE_LABELS = {
    "eastmoney": "东方财富",
    "sina": "新浪",
    "tencent": "腾讯",
}


def normalize_data_source(source: str) -> str:
    normalized = source.strip().lower()
    if normalized not in _DATA_SOURCE_LABELS:
        raise ValueError(f"不支持的数据源：{source}")
    return normalized


def get_data_source_label(source: str) -> str:
    return _DATA_SOURCE_LABELS[normalize_data_source(source)]


def _download_single(ticker: str, source: str = "eastmoney", market: str = "a_share") -> pd.DataFrame | None:
    if market == "us":
        return _download_from_yfinance(ticker)
    selected = normalize_data_source(source)
    loaders = {
        "eastmoney": _download_from_eastmoney,
        "sina": _download_from_sina,
        "tencent": _download_from_tencent,
    }
    if selected == "eastmoney":
        return loaders[selected](ticker)
    try:
        return loaders[selected](ticker)
    except Exception as exc:
        logger.debug("数据源 %s 获取 %s 失败：%s", get_data_source_label(selected), ticker, exc)
        return None


def download_ticker(ticker: str, force: bool = False, source: str = "eastmoney", market: str = "a_share") -> pd.DataFrame | None:
    """
    Get OHLCV data for *ticker*.
    - If cached data exists, load it and download only the missing tail.
    - If *force* is True, re-download everything.
    """
    selected = normalize_data_source(source)
    cache_key = f"{ticker}__us" if market == "us" else f"{ticker}__{selected}"
    if force:
        df = _download_single(ticker, selected, market)
        if df is not None:
            _save_cache(cache_key, df)
        return df

    cached = _load_cache(cache_key)
    if cached is None:
        df = _download_single(ticker, selected, market)
        if df is not None:
            _save_cache(cache_key, df)
        return df

    # Incremental update: download only from the last cached date
    last_date = cached.index.max()
    today = datetime.now()

    # Normalise to naive datetime for comparison
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.to_pydatetime()
    if last_date.tzinfo is not None:
        last_date = last_date.replace(tzinfo=None)

    # Guard: if last_date is somehow in the future, skip update
    if (today - last_date).days <= 1:
        return cached  # already up-to-date

    try:
        full_df = _download_single(ticker, selected, market)
        new_df = full_df.loc[full_df.index > pd.Timestamp(last_date)] if full_df is not None else None
        if new_df is not None and not new_df.empty:
            new_df = new_df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            new_df = new_df[["Open", "High", "Low", "Close", "Volume"]]
            new_df = new_df.dropna(subset=["Close"])
            # Strip timezone from new data to match cached
            try:
                idx = pd.DatetimeIndex(new_df.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                new_df.index = idx
            except Exception:
                pass
            if not new_df.empty:
                combined = pd.concat([cached, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                _save_cache(cache_key, combined)
                return combined
    except Exception as exc:
        logger.debug("Incremental update failed for %s: %s — using cache as-is.", ticker, exc)

    return cached


def download_batch(
    tickers: list[TickerInfo],
    desc: str = "Downloading",
    force: bool = False,
    source: str = "eastmoney",
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

    # Build ticker->market map for routing downloads
    market_map = {t.ticker: t.market for t in tickers}

    # Single-threaded download with inter-request pause
    if DOWNLOAD_THREADS <= 1:
        for sym in tqdm(symbols, desc=desc, unit="ticker"):
            try:
                df = download_ticker(sym, force=force, source=source,
                                     market=market_map.get(sym, "a_share"))
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
                pool.submit(download_ticker, sym, force, source, market_map.get(sym, "a_share")): sym
                for sym in symbols
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
    return None
