"""
social_sentiment.py — Free-tier social sentiment and short squeeze detection

Three data sources, all free, no API keys required:
  1. StockTwits stream      — bullish/bearish crowd sentiment per ticker
  2. NASDAQ short interest  — days-to-cover, short float proxy
  3. FINRA daily short vol  — intraday short pressure ratio (CNMSshvol)

Public API (imported by scanner.py and predictions.py):
  get_social_score(ticker)                          → float  -10 to +10
  get_short_score(ticker, rvol=1.0, price_trend=0)  → float    0 to +15
  get_social_batch(tickers)                         → dict per ticker
  get_status_summary()                              → dict for logging

Design notes:
  - All network calls are fire-and-forget; failures return neutral values.
  - One threading.Lock guards all three caches (double-checked locking).
  - This module is imported lazily inside try/except in scanner.py and
    predictions.py, so it must never raise on import.
"""

from __future__ import annotations

import io
import threading
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

__all__ = [
    "get_social_score",
    "get_short_score",
    "get_social_batch",
    "get_status_summary",
]

# ── Cache TTLs (seconds) ──────────────────────────────────────────────────────

_SOCIAL_TTL = 30 * 60        # 30 minutes — StockTwits
_SHORT_TTL  = 4  * 60 * 60   # 4 hours    — NASDAQ short interest
_FINRA_TTL  = 24 * 60 * 60   # 24 hours   — FINRA file (one download per day)

# ── Module-level caches ───────────────────────────────────────────────────────
# Keyed by uppercase ticker symbol.
# Each entry always carries a "ts" (float) timestamp for TTL checks.

# { "AAPL": {"score": float, "bull_pct": float, "ts": float}, ... }
_social_cache: Dict[str, dict] = {}

# { "AAPL": {"dtc": float, "short_float": float, "ts": float}, ... }
_short_cache: Dict[str, dict] = {}

# { "date": "YYYYMMDD",
#   "data": { "AAPL": {"short_ratio": float}, ... },
#   "ts": float }
_finra_cache: Dict = {}

# Single lock guards all three caches — simple and avoids lock-ordering bugs.
_cache_lock = threading.Lock()

# ── Shared HTTP session ───────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible)",
    "Accept-Encoding": "gzip, deflate",
})

_REQUEST_TIMEOUT = 12  # seconds per request
_MAX_RETRIES     = 1   # one silent retry on connection errors (not 4xx/5xx)

# ── Internal HTTP helper ──────────────────────────────────────────────────────


def _get(
    url: str,
    params: Optional[dict] = None,
    extra_headers: Optional[dict] = None,
) -> Optional[requests.Response]:
    """
    HTTP GET with timeout, one retry on transient errors, and full error
    suppression.  Returns a Response (raise_for_status already called) or None.
    """
    headers: dict = {}
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(
                url,
                params=params,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as exc:
            # Transient — worth one retry
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(0.5)
        except requests.exceptions.HTTPError as exc:
            # 4xx / 5xx — don't retry; data just isn't available
            print(f"  [social_sentiment] HTTP {exc.response.status_code} {url}: {exc}")
            return None
        except requests.exceptions.Timeout as exc:
            print(f"  [social_sentiment] Timeout {url}: {exc}")
            return None
        except Exception as exc:
            print(f"  [social_sentiment] Unexpected error {url}: {exc}")
            return None

    print(f"  [social_sentiment] Connection error {url}: {last_exc}")
    return None


# ── Trading-day helper ────────────────────────────────────────────────────────


def _prev_trading_day(ref: Optional[date] = None) -> date:
    """
    Return the most recent completed *trading* weekday strictly before `ref`
    (default: today).  Used for FINRA URLs because FINRA publishes T-1 data.
    """
    d = (ref or date.today()) - timedelta(days=1)
    while d.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
        d -= timedelta(days=1)
    return d


def _date_str(d: date) -> str:
    """Format date as YYYYMMDD for FINRA URL."""
    return d.strftime("%Y%m%d")


# ── StockTwits ────────────────────────────────────────────────────────────────

_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

_SOCIAL_NEUTRAL: Dict = {"score": 0.0, "bull_pct": 0.5}


def _fetch_stocktwits(ticker: str) -> Dict:
    """
    Fetch StockTwits stream for *ticker* and compute a sentiment score.

    Returns {"score": float [-10,+10], "bull_pct": float [0,1]}.
    Falls back to _SOCIAL_NEUTRAL on any failure or missing data.

    Scoring:
      raw_score = (bull_ratio - 0.5) × 20
      × 1.3 if ≥ 20 tagged messages (strong-signal boost)
      × 1.1 if ≥ 10 tagged messages (moderate-signal boost)
      clamped to [-10, +10]
    """
    url  = _STOCKTWITS_URL.format(ticker=ticker)
    resp = _get(url)
    if resp is None:
        return _SOCIAL_NEUTRAL.copy()

    try:
        payload = resp.json()
    except ValueError as exc:
        print(f"  [social_sentiment] StockTwits JSON parse error {ticker}: {exc}")
        return _SOCIAL_NEUTRAL.copy()

    messages = payload.get("messages") or []
    if not messages:
        return _SOCIAL_NEUTRAL.copy()

    bull_count = 0
    bear_count = 0

    for msg in messages:
        sentiment_obj = (msg.get("entities") or {}).get("sentiment")
        if not isinstance(sentiment_obj, dict):
            continue
        basic = sentiment_obj.get("basic", "")
        if basic == "Bullish":
            bull_count += 1
        elif basic == "Bearish":
            bear_count += 1

    total_tagged = bull_count + bear_count
    if total_tagged == 0:
        return _SOCIAL_NEUTRAL.copy()

    bull_ratio = bull_count / total_tagged
    raw_score  = (bull_ratio - 0.5) * 20.0

    # Confidence multiplier based on tagged-message sample size
    if total_tagged >= 20:
        raw_score *= 1.3
    elif total_tagged >= 10:
        raw_score *= 1.1

    score = max(-10.0, min(10.0, raw_score))
    return {"score": round(score, 3), "bull_pct": round(bull_ratio, 4)}


def _get_social_cached(ticker: str) -> Dict:
    """Return StockTwits result from cache, refreshing if stale."""
    ticker = ticker.upper()
    now    = time.monotonic()

    # Fast path — no lock needed for reads when the entry is fresh
    entry = _social_cache.get(ticker)
    if entry is not None and (now - entry["ts"]) < _SOCIAL_TTL:
        return entry

    with _cache_lock:
        # Re-check inside the lock (double-checked locking)
        entry = _social_cache.get(ticker)
        if entry is not None and (now - entry["ts"]) < _SOCIAL_TTL:
            return entry

        data  = _fetch_stocktwits(ticker)
        entry = {**data, "ts": now}
        _social_cache[ticker] = entry

    return entry


# ── NASDAQ Short Interest ─────────────────────────────────────────────────────

_NASDAQ_SHORT_URL    = "https://api.nasdaq.com/api/quote/{ticker}/short-interest"
_NASDAQ_SHORT_PARAMS = {"assetClass": "stocks"}
_NASDAQ_SHORT_HDR    = {
    "Accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (compatible)",
    "Referer":    "https://www.nasdaq.com/",
}

_SHORT_NEUTRAL: Dict = {"dtc": 0.0, "short_float": 0.0}


def _dtc_to_base_score(dtc: float) -> float:
    """
    Map days-to-cover to a 0–15 base short-squeeze score.

    Tiers:
      dtc < 1  → 0   (negligible short interest)
      1 ≤ dtc < 3  → 3   (light)
      3 ≤ dtc < 5  → 6   (moderate)
      5 ≤ dtc ≤ 10 → 10  (heavy)
      dtc > 10 → 15  (extreme — classic squeeze setup)
    """
    if dtc < 1.0:
        return 0.0
    if dtc < 3.0:
        return 3.0
    if dtc < 5.0:
        return 6.0
    if dtc <= 10.0:
        return 10.0
    return 15.0


def _safe_float(val: object, fallback: float = 0.0) -> float:
    """Parse a possibly comma-formatted numeric string; return fallback on failure."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return fallback


def _fetch_nasdaq_short(ticker: str) -> Dict:
    """
    Fetch NASDAQ short interest data for *ticker*.

    Returns {"dtc": float, "short_float": float}.
    Note: "short_float" here is short_interest / avg_daily_volume, a proxy
    for days-to-cover expressed differently.  True short-float % requires
    shares-outstanding data not available from this endpoint.
    """
    url  = _NASDAQ_SHORT_URL.format(ticker=ticker)
    resp = _get(url, params=_NASDAQ_SHORT_PARAMS, extra_headers=_NASDAQ_SHORT_HDR)
    if resp is None:
        return _SHORT_NEUTRAL.copy()

    try:
        payload = resp.json()
    except ValueError as exc:
        print(f"  [social_sentiment] NASDAQ JSON parse error {ticker}: {exc}")
        return _SHORT_NEUTRAL.copy()

    try:
        rows: list = (
            (payload.get("data") or {})
            .get("shortInterestTable", {})
            .get("rows") or []
        )
        if not rows:
            return _SHORT_NEUTRAL.copy()

        row       = rows[0]
        dtc       = _safe_float(row.get("daysToCover",       0))
        short_int = _safe_float(row.get("shortInterest",     0))
        avg_vol   = _safe_float(row.get("avgDailyShareVolume", 1)) or 1.0

        short_float = short_int / avg_vol if avg_vol > 0 else 0.0
        return {"dtc": round(dtc, 2), "short_float": round(short_float, 4)}

    except Exception as exc:
        print(f"  [social_sentiment] NASDAQ parse error {ticker}: {exc}")
        return _SHORT_NEUTRAL.copy()


def _get_short_cached(ticker: str) -> Dict:
    """Return NASDAQ short interest result from cache, refreshing if stale."""
    ticker = ticker.upper()
    now    = time.monotonic()

    entry = _short_cache.get(ticker)
    if entry is not None and (now - entry["ts"]) < _SHORT_TTL:
        return entry

    with _cache_lock:
        entry = _short_cache.get(ticker)
        if entry is not None and (now - entry["ts"]) < _SHORT_TTL:
            return entry

        data  = _fetch_nasdaq_short(ticker)
        entry = {**data, "ts": now}
        _short_cache[ticker] = entry

    return entry


# ── FINRA Short Volume ────────────────────────────────────────────────────────

_FINRA_URL_TMPL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
)

# Expected header columns (0-based):
#   0=Date  1=Symbol  2=ShortVolume  3=ShortExemptVolume  4=TotalVolume  5=Market
_FINRA_COL_SYMBOL = 1
_FINRA_COL_SHORT  = 2
_FINRA_COL_TOTAL  = 4


def _fetch_finra_file(day_str: str) -> Dict[str, Dict[str, float]]:
    """
    Download and parse FINRA CNMSshvol pipe-delimited short volume file for
    *day_str* (YYYYMMDD).

    Multiple rows per symbol (one per market centre) are aggregated before
    computing the final short_ratio = ShortVolume / TotalVolume.

    Returns {SYMBOL: {"short_ratio": float}} or {} on failure.
    """
    url  = _FINRA_URL_TMPL.format(date=day_str)
    resp = _get(url)
    if resp is None:
        return {}

    # Raw accumulators: {symbol: [total_short_vol, total_vol]}
    accum: Dict[str, List[int]] = {}

    try:
        reader = io.StringIO(resp.text)

        for lineno, line in enumerate(reader):
            line = line.strip()
            if not line:
                continue
            # Skip the header row (starts with "Date") regardless of position
            if line.startswith("Date"):
                continue

            parts = line.split("|")
            if len(parts) <= _FINRA_COL_TOTAL:
                continue

            try:
                symbol    = parts[_FINRA_COL_SYMBOL].strip().upper()
                short_vol = int(parts[_FINRA_COL_SHORT].strip())
                total_vol = int(parts[_FINRA_COL_TOTAL].strip())
            except (ValueError, IndexError):
                continue

            if not symbol or total_vol <= 0:
                continue

            if symbol in accum:
                accum[symbol][0] += short_vol
                accum[symbol][1] += total_vol
            else:
                accum[symbol] = [short_vol, total_vol]

    except Exception as exc:
        print(f"  [social_sentiment] FINRA parse error {day_str}: {exc}")
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for sym, (sv, tv) in accum.items():
        if tv > 0:
            result[sym] = {"short_ratio": round(sv / tv, 4)}

    return result


def _get_finra_cached() -> Dict[str, Dict[str, float]]:
    """
    Return the FINRA short-ratio lookup dict for the previous trading day,
    downloading at most once per 24-hour period.

    Returns {} on failure.
    """
    now     = time.monotonic()
    day_str = _date_str(_prev_trading_day())

    # Fast read — only lock if we need to refresh
    fc = _finra_cache
    if (
        fc.get("date") == day_str
        and fc.get("data") is not None
        and (now - fc.get("ts", 0.0)) < _FINRA_TTL
    ):
        return fc["data"]  # type: ignore[return-value]

    with _cache_lock:
        # Double-check inside lock
        fc = _finra_cache
        if (
            fc.get("date") == day_str
            and fc.get("data") is not None
            and (now - fc.get("ts", 0.0)) < _FINRA_TTL
        ):
            return fc["data"]  # type: ignore[return-value]

        data = _fetch_finra_file(day_str)
        _finra_cache["date"] = day_str
        _finra_cache["data"] = data
        _finra_cache["ts"]   = now

    return _finra_cache["data"]  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────


def get_social_score(ticker: str) -> float:
    """
    Return crowd sentiment score for *ticker* sourced from StockTwits.

    Range: -10.0 (very bearish) → +10.0 (very bullish).
    Returns 0.0 when data is unavailable or an error occurs.
    """
    try:
        return _get_social_cached(ticker).get("score", 0.0)
    except Exception as exc:
        print(f"  [social_sentiment] get_social_score error {ticker}: {exc}")
        return 0.0


def get_short_score(
    ticker: str,
    rvol: float = 1.0,
    price_trend: int = 0,
) -> float:
    """
    Return short-squeeze potential score for *ticker*.

    Range: 0.0 (no squeeze pressure) → 15.0 (extreme squeeze potential).

    Parameters
    ----------
    ticker      : stock symbol (case-insensitive)
    rvol        : relative volume vs 20-day average (1.0 = average day)
    price_trend : +1 uptrend, 0 flat/unknown, -1 downtrend

    Scoring breakdown
    -----------------
    Base score (DTC tiers, NASDAQ):
      dtc < 1          →  0
      1 ≤ dtc < 3      →  3
      3 ≤ dtc < 5      →  6
      5 ≤ dtc ≤ 10     → 10
      dtc > 10         → 15

    FINRA intraday pressure bonus (+3):
      short_ratio > 0.55 AND rvol > 2.0
      → shorts are being caught on heavy volume, squeeze acceleration likely

    The DTC-based NASDAQ score already encodes the squeeze potential level.
    The rvol / price_trend gate for the FINRA bonus ensures we only add the
    extra weight when there is *both* intraday evidence (FINRA) AND elevated
    volume confirming the squeeze is actively underway.

    All scores hard-capped at 15.0.
    """
    try:
        ticker_up  = ticker.upper()
        short_data = _get_short_cached(ticker_up)
        dtc        = short_data.get("dtc", 0.0)
        score      = _dtc_to_base_score(dtc)

        # FINRA intraday short pressure bonus:
        # Only applies when volume is elevated AND price not falling
        # (i.e., shorts are being squeezed, not piling in on a down move).
        if rvol >= 2.0 and price_trend >= 0:
            finra_data  = _get_finra_cached()
            sym_entry   = finra_data.get(ticker_up, {})
            short_ratio = sym_entry.get("short_ratio", 0.0)
            if short_ratio > 0.55:
                score += 3.0

        return round(min(score, 15.0), 3)

    except Exception as exc:
        print(f"  [social_sentiment] get_short_score error {ticker}: {exc}")
        return 0.0


def get_social_batch(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch social + short data for a list of tickers in one call.

    The FINRA file is downloaded once and shared across all tickers.

    Returns
    -------
    {
        "AAPL": {
            "social":   float,  # -10 to +10
            "short":    float,  #   0 to +15
            "squeeze":  bool,   # short >= 8 AND social >= 0
            "bull_pct": float,  # 0.0–1.0 fraction bullish on StockTwits
            "dtc":      float,  # days-to-cover from NASDAQ
        },
        ...
    }

    Missing / errored tickers return safe neutral defaults rather than raising.
    """
    result: Dict[str, Dict] = {}

    # Pre-warm the FINRA cache once before the per-ticker loop so every call
    # to get_short_score() below reuses the already-cached data.
    try:
        _get_finra_cached()
    except Exception as exc:
        print(f"  [social_sentiment] FINRA pre-warm error: {exc}")

    for raw_ticker in tickers:
        ticker = raw_ticker.upper()
        try:
            social_entry = _get_social_cached(ticker)
            short_entry  = _get_short_cached(ticker)

            social_score = float(social_entry.get("score",    0.0))
            bull_pct     = float(social_entry.get("bull_pct", 0.5))
            dtc          = float(short_entry.get("dtc",       0.0))

            # Use default rvol/price_trend; callers that have live bar data
            # should call get_short_score(ticker, rvol=..., price_trend=...)
            # directly for a more precise score.
            short_score = get_short_score(ticker)

            result[ticker] = {
                "social":   social_score,
                "short":    short_score,
                "squeeze":  short_score >= 8.0 and social_score >= 0.0,
                "bull_pct": bull_pct,
                "dtc":      dtc,
            }

        except Exception as exc:
            print(f"  [social_sentiment] get_social_batch error {ticker}: {exc}")
            result[ticker] = {
                "social":   0.0,
                "short":    0.0,
                "squeeze":  False,
                "bull_pct": 0.5,
                "dtc":      0.0,
            }

    return result


def get_status_summary() -> Dict:
    """
    Return a snapshot of module cache health for logging and diagnostics.

    Keys
    ----
    social_cached    int   — number of tickers in the StockTwits cache
    short_cached     int   — number of tickers in the NASDAQ short cache
    finra_loaded     bool  — True if FINRA data is in memory
    finra_date       str   — YYYYMMDD of the loaded FINRA file ("" if none)
    finra_symbols    int   — number of symbols in the loaded FINRA data
    finra_ttl_secs   float — seconds remaining until FINRA refresh (0 if expired)
    """
    now = time.monotonic()
    with _cache_lock:
        finra_data    = _finra_cache.get("data") or {}
        finra_loaded  = bool(finra_data)
        finra_date    = str(_finra_cache.get("date", ""))
        finra_symbols = len(finra_data)
        finra_age     = now - _finra_cache.get("ts", 0.0)
        finra_ttl     = max(0.0, _FINRA_TTL - finra_age)
        social_count  = len(_social_cache)
        short_count   = len(_short_cache)

    return {
        "social_cached":  social_count,
        "short_cached":   short_count,
        "finra_loaded":   finra_loaded,
        "finra_date":     finra_date,
        "finra_symbols":  finra_symbols,
        "finra_ttl_secs": round(finra_ttl, 1),
    }
