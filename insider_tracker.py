"""
insider_tracker.py — SEC EDGAR Form 4 insider purchase tracker

Fetches Form 4 filings (corporate insiders buying their own company's stock)
from the SEC EDGAR full-text search API. No API key required; SEC asks only
for a descriptive User-Agent header.

Results are cached for 6 hours so the free endpoint is never hammered.

Architecture: one broad batch fetch covers all recent Form 4 filings at once
rather than per-ticker queries. Ticker symbols are extracted from the entity
names in the response (e.g. "Apple Inc. (AAPL)").

Exports used by the rest of the bot:
  get_insider_scores(days=30)             → {ticker: score 0–20}
  get_insider_tickers(days=30, min_score=3) → [ticker, ...]  (top 15 max)
  get_status_summary()                    → dict for logging/dashboard

Score meaning:
  0      = no recent insider purchase activity
  1–5    = light interest (single director, older filing)
  6–12   = moderate interest (multiple insiders or senior officer)
  13–20  = strong signal (CEO/CFO + multiple insiders, very recent)
"""

import re
import time
import threading
from datetime import datetime, timedelta

import pytz
import requests

# ── Constants ─────────────────────────────────────────────────────────────────

NY        = pytz.timezone("America/New_York")
CACHE_TTL = 6 * 3600   # 6 hours, matches politician_tracker.py

# SEC requires a descriptive User-Agent; bots that omit it get rate-limited.
_SEC_HEADERS = {"User-Agent": "trading-bot contact@example.com"}

# Base EDGAR full-text search endpoint for Form 4 filings.
# We pull up to 200 hits per call (EDGAR caps a single page at 200).
# No q= filter is used so we get all Form 4s for the window, then
# filter for purchases in _parse_hit().
_EFTS_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?forms=4"
    "&dateRange=custom"
    "&startdt={start}"
    "&enddt={end}"
    "&hits.hits.total.value=true"
    "&from=0"
    "&size=200"
)

# Regex to extract a NYSE/NASDAQ ticker symbol from strings like
# "Apple Inc. (AAPL)" or "Apple Inc. (AAPL) | Tim Cook"
_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")

# Word-boundary regex to find standalone 'P' (purchase) transaction code.
# Form 4 XML uses single letters: 'P' = open-market purchase, 'S' = sale,
# 'A' = grant, etc. The EFTS _source text sometimes surfaces these.
_PURCHASE_CODE_RE = re.compile(r"\bP\b")
_SALE_CODE_RE     = re.compile(r"\bS\b")

# Officer-title keywords — matched using word-boundary regex to prevent
# substring collisions (e.g. "cto" inside "director").
_CEO_RE  = re.compile(r"\b(chief\s+executive|ceo|president)\b", re.I)
_CFO_RE  = re.compile(
    r"\b(chief\s+financial|cfo|chief\s+operating|coo|"
    r"chief\s+technology|cto|chief\s+revenue)\b",
    re.I,
)
_DIR_RE  = re.compile(r"\bdirector\b", re.I)

# Common false-positive ticker patterns to discard (legal suffixes, common
# abbreviations that appear in parentheses but are not exchange symbols).
_TICKER_BLOCKLIST = frozenset({
    "INC", "LLC", "LTD", "CORP", "CO", "NA", "LP", "PLC",
    "ETF", "ADR", "USA", "US", "THE", "AND", "OF", "FOR",
    "NEW", "OLD", "II", "III", "IV",
})

# Scoring constants
_BASE_PTS     = 3   # points per qualifying filing
_CEO_BONUS    = 4   # additional points if filer is CEO / President
_CFO_BONUS    = 3   # additional points for CFO / COO / CTO level
_DIR_BONUS    = 2   # additional points for board directors
_DIVERSITY_PT = 2   # additional points per unique insider beyond the first
_MAX_SCORE    = 20

# ── Module-level cache & lock ──────────────────────────────────────────────────

_lock        = threading.Lock()
_raw_cache   = {"data": None, "ts": 0}   # list[dict] of parsed filings
_score_cache = {"data": None, "ts": 0}   # {ticker: float}

# ── Internal helpers ───────────────────────────────────────────────────────────

def _date_window(days: int) -> tuple[str, str]:
    """Return (start_str, end_str) in YYYY-MM-DD covering the last *days* days."""
    today = datetime.now(pytz.utc).date()
    start = today - timedelta(days=days)
    return str(start), str(today)


def _extract_ticker(text: str) -> str | None:
    """
    Pull a NYSE/NASDAQ ticker from a string that may contain it in parentheses,
    e.g. 'Apple Inc. (AAPL)' → 'AAPL'.

    Returns None if no plausible symbol is found.  Filters out legal-suffix
    false positives via the block-list.
    """
    if not text:
        return None
    m = _TICKER_RE.search(text)
    if m:
        candidate = m.group(1)
        if candidate not in _TICKER_BLOCKLIST:
            return candidate
    return None


def _classify_title(title_text: str) -> str:
    """
    Return 'ceo', 'cfo', 'director', or '' based on officer title keywords.
    Uses word-boundary regex to avoid substring false positives (e.g. the
    substring 'cto' inside 'director').

    Return values:
      'ceo'      → CEO, President  (+4 bonus)
      'cfo'      → CFO, COO, CTO   (+3 bonus)
      'director' → Board director  (+2 bonus)
      ''         → unknown / no match
    """
    # CEO check first: highest priority
    if _CEO_RE.search(title_text):
        return "ceo"
    # Director check BEFORE CFO/CTO: 'director' contains 'cto' as substring
    if _DIR_RE.search(title_text):
        return "director"
    # CFO/COO/CTO last (after director to avoid the 'cto'⊂'director' trap)
    if _CFO_RE.search(title_text):
        return "cfo"
    return ""


def _looks_like_purchase(source: dict) -> bool:
    """
    Heuristic: decide whether this Form 4 hit is likely an open-market
    purchase rather than a sale or award.

    EDGAR's EFTS search endpoint does not reliably surface the raw XML
    transaction code in the indexed _source dict.  We therefore use a
    best-effort approach:

    1. If the source text contains a standalone 'S' sale code, reject it.
    2. If the source text contains a standalone 'P' purchase code, accept it.
    3. Otherwise accept: Form 4s with no detectable code are ambiguous; we
       include them to avoid false negatives, relying on the score cap (20)
       to contain damage from any isolated misclassification.

    Note: Form 4/A (amendments) are included; they may amend a prior purchase.
    """
    # Gather all string fields into one text blob for pattern matching
    text_blob = " ".join(
        str(v) for v in source.values() if isinstance(v, str)
    )

    # Explicit sale signal → discard
    if _SALE_CODE_RE.search(text_blob) and not _PURCHASE_CODE_RE.search(text_blob):
        return False

    return True


def _parse_hit(hit: dict) -> dict | None:
    """
    Parse one EFTS hit dict into a normalised filing record, or return None
    if the hit should be skipped (no ticker, not a purchase, missing date).

    Returned dict keys:
      ticker, filer_name, title_class, file_date, period_of_report
    """
    try:
        src = hit.get("_source") or {}

        # ── Filter out non-purchases ──────────────────────────────────────
        if not _looks_like_purchase(src):
            return None

        # ── Ticker extraction ─────────────────────────────────────────────
        # display_names is typically "Company Name (TICK) | Person Name"
        display_names = src.get("display_names") or ""
        if isinstance(display_names, list):
            display_names = " ".join(display_names)

        ticker = _extract_ticker(display_names)

        if not ticker:
            entity_name = src.get("entity_name") or ""
            if isinstance(entity_name, list):
                entity_name = " ".join(entity_name)
            ticker = _extract_ticker(entity_name)

        if not ticker:
            return None   # cannot attribute this filing without a symbol

        # ── Filer name ────────────────────────────────────────────────────
        # display_names format: "Company (TICK) | Filer Name | ..."
        # First segment is the company; the rest are filers / signatories.
        if "|" in display_names:
            parts = [p.strip() for p in display_names.split("|")]
            filer_name = " | ".join(parts[1:]) if len(parts) > 1 else parts[0]
        else:
            filer_name = display_names or src.get("entity_name", "") or "Unknown"

        # ── Officer title classification ──────────────────────────────────
        # Scan all string fields; stop at the first recognisable title.
        title_class = ""
        for field_val in src.values():
            if isinstance(field_val, str):
                cls = _classify_title(field_val)
                if cls:
                    title_class = cls
                    break   # highest-priority match already applied by _classify_title

        # ── Dates ─────────────────────────────────────────────────────────
        file_date = (src.get("file_date") or src.get("period_of_report") or "")[:10]
        period    = (src.get("period_of_report") or file_date or "")[:10]

        if not file_date or len(file_date) < 10:
            return None

        return {
            "ticker":           ticker,
            "filer_name":       filer_name,
            "title_class":      title_class,
            "file_date":        file_date,
            "period_of_report": period,
        }

    except Exception as exc:
        print(f"Insider tracker: parse error on hit — {exc}")
        return None


def _fetch_filings(days: int) -> list[dict]:
    """
    Fetch up to 200 Form 4 filings from EDGAR EFTS for the given date window.

    Returns a list of normalised filing dicts as produced by _parse_hit().
    Any HTTP or JSON errors are caught and an empty list is returned so that
    callers always receive a valid (possibly empty) result.
    """
    start, end = _date_window(days)
    url = _EFTS_URL.format(start=start, end=end)

    try:
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        print(f"Insider tracker: EFTS HTTP error {exc.response.status_code} — {exc}")
        return []
    except requests.exceptions.RequestException as exc:
        print(f"Insider tracker: EFTS request error — {exc}")
        return []
    except ValueError as exc:
        print(f"Insider tracker: EFTS JSON parse error — {exc}")
        return []

    try:
        hits = data.get("hits", {}).get("hits", []) or []
    except Exception as exc:
        print(f"Insider tracker: unexpected EFTS response shape — {exc}")
        return []

    if not hits:
        total = data.get("hits", {}).get("total", {})
        print(f"Insider tracker: no hits returned from EFTS (total={total}, "
              f"window {start}→{end})")
        return []

    results = [r for hit in hits if (r := _parse_hit(hit)) is not None]

    print(f"Insider tracker: {len(results)} Form 4 purchase filings parsed "
          f"({len(hits)} raw hits, window {start}→{end})")
    return results


def _load_raw(days: int) -> list[dict]:
    """
    Return the raw filing list, refreshing the 6-hour cache when stale.

    Uses a double-checked locking pattern:
      - First check (inside lock) returns immediately if cache is warm.
      - HTTP fetch happens *outside* the lock so other threads are not blocked.
      - Second check (inside lock) avoids a redundant cache write if another
        thread populated the cache during our fetch.
    """
    # Fast path
    with _lock:
        if (_raw_cache["data"] is not None
                and time.time() - _raw_cache["ts"] < CACHE_TTL):
            return _raw_cache["data"]

    # Slow path: fetch outside the lock
    filings = _fetch_filings(days)

    with _lock:
        # Re-check: skip write if another thread beat us to it
        if (_raw_cache["data"] is None
                or time.time() - _raw_cache["ts"] >= CACHE_TTL):
            _raw_cache["data"] = filings
            _raw_cache["ts"]   = time.time()
        else:
            filings = _raw_cache["data"]

    return filings


# ── Public API ─────────────────────────────────────────────────────────────────

def get_insider_scores(days: int = 30) -> dict[str, float]:
    """
    Return {ticker: score 0–20} for all tickers with recent Form 4 insider
    purchases detected in the last *days* days.

    Scoring per filing:
      - 3 pts base
      - +4 if the filer is CEO or President
      - +3 if the filer is CFO, COO, or CTO level
      - +2 if the filer is a Director

    Adjustments across filings for the same ticker:
      - Recency decay applied per filing:
          ≤ 7 days ago  → 1.0×
          8–14 days ago → 0.7×
          15–30 days    → 0.4×
      - +2 diversity bonus per unique insider beyond the first

    Final score capped at 20. Results cached for 6 hours.
    """
    # Fast path: score cache is warm
    with _lock:
        if (_score_cache["data"] is not None
                and time.time() - _score_cache["ts"] < CACHE_TTL):
            return _score_cache["data"]

    filings = _load_raw(days)
    today   = datetime.now(pytz.utc).date()
    cutoff  = today - timedelta(days=days)

    raw_scores:    dict[str, float]       = {}
    insider_names: dict[str, set[str]]    = {}

    for filing in filings:
        try:
            ticker      = filing["ticker"]
            file_date   = filing["file_date"]
            filer_name  = filing.get("filer_name") or "unknown"
            title_class = filing.get("title_class") or ""

            # Parse the filing date
            try:
                fd = datetime.strptime(file_date, "%Y-%m-%d").date()
            except ValueError:
                continue

            if fd < cutoff:
                continue

            # Recency decay
            days_ago = (today - fd).days
            if   days_ago <= 7:  recency = 1.0
            elif days_ago <= 14: recency = 0.7
            else:                recency = 0.4   # 15–30 days

            # Base score + title bonus
            pts = _BASE_PTS
            if   title_class == "ceo":      pts += _CEO_BONUS
            elif title_class == "cfo":      pts += _CFO_BONUS
            elif title_class == "director": pts += _DIR_BONUS

            raw_scores[ticker]  = raw_scores.get(ticker, 0.0) + pts * recency
            insider_names.setdefault(ticker, set()).add(filer_name.lower())

        except Exception as exc:
            print(f"Insider tracker: scoring error — {exc}")
            continue

    # Apply diversity bonus: +2 per unique insider beyond the first
    result: dict[str, float] = {}
    for ticker, raw in raw_scores.items():
        n_insiders  = len(insider_names.get(ticker, set()))
        diversity   = max(0, n_insiders - 1) * _DIVERSITY_PT
        final_score = min(_MAX_SCORE, raw + diversity)
        result[ticker] = round(final_score, 1)

    with _lock:
        # Re-check before writing: use any result another thread produced first
        if (_score_cache["data"] is None
                or time.time() - _score_cache["ts"] >= CACHE_TTL):
            _score_cache["data"] = result
            _score_cache["ts"]   = time.time()
        else:
            result = _score_cache["data"]

    if result:
        top = sorted(result.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"Insider tracker top tickers: {top}")

    return result


def get_insider_tickers(days: int = 30, min_score: float = 3) -> list[str]:
    """
    Return tickers with meaningful recent insider purchase activity, ordered
    by score descending. Returns at most 15 tickers.

    Parameters
    ----------
    days      : look-back window in days (default 30)
    min_score : minimum score threshold to include a ticker (default 3)
    """
    scores = get_insider_scores(days=days)
    ranked = [(t, s) for t, s in scores.items() if s >= min_score]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:15]]


def get_status_summary() -> dict:
    """
    Return a brief status dict suitable for dashboard or structured logging.

    Keys:
      filings_loaded   : number of Form 4 filings currently in raw cache
      tickers_tracked  : number of distinct tickers with a non-zero score
      top_tickers      : list of (ticker, score) for the top 5 tickers
      cache_age_mins   : age of the raw cache in minutes, or None if empty
      source           : data source identifier string
    """
    with _lock:
        raw_data = _raw_cache["data"]
        raw_ts   = _raw_cache["ts"]
        scores   = _score_cache.get("data") or {}

    return {
        "filings_loaded":  len(raw_data) if raw_data is not None else 0,
        "tickers_tracked": len(scores),
        "top_tickers":     sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:5],
        "cache_age_mins":  round((time.time() - raw_ts) / 60, 1) if raw_ts else None,
        "source":          "SEC EDGAR EFTS (Form 4)",
    }
