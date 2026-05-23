"""
politician_tracker.py — Congressional STOCK Act trade tracker

Fetches senator purchase disclosures from Senate Stock Watcher (no API key,
no auth required). Results are cached for 6 hours so the free endpoint is
never hammered.

Exports used by the rest of the bot:
  get_ticker_scores(days=90) → {ticker: score 0–25}
  get_politician_tickers(days=60, min_score=3) → [ticker, ...]

Score meaning:
  0     = no recent congressional activity
  1–5   = light interest (small amounts, 1 politician)
  6–12  = moderate interest (multiple buys or larger amounts)
  13–25 = strong interest (several politicians, large amounts, recent)
"""

import os, time, threading
import requests
from datetime import datetime, timedelta
import pytz

NY = pytz.timezone("America/New_York")

# ── Endpoints (tried in order) ────────────────────────────────
_SOURCES = [
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    "https://senatestockwatcher.com/api/trades.json",
]

# Optional: Financial Modeling Prep covers both chambers (free tier available)
# Set FMP_API_KEY env var to enable. Get a free key at financialmodelingprep.com
FMP_KEY       = os.environ.get("FMP_API_KEY")
FMP_SENATE    = "https://financialmodelingprep.com/api/v4/senate-disclosure"
FMP_HOUSE     = "https://financialmodelingprep.com/api/v4/house-disclosure"

CACHE_TTL     = 6 * 3600   # refresh every 6 hours

_lock         = threading.Lock()
_raw_cache    = {"data": None, "ts": 0, "source": None}
_score_cache  = {"data": None, "ts": 0}

# ── Amount range parser ───────────────────────────────────────

_AMOUNT_MAP = [
    ("$1,000,001", 6),
    ("$500,001",   5),
    ("$100,001",   4),
    ("$50,001",    3),
    ("$15,001",    2),
    ("$1,001",     1),
]

def _parse_amount(amount_str):
    """'$15,001 - $50,000' → integer score 1–6 reflecting trade size."""
    if not amount_str:
        return 1
    for marker, score in _AMOUNT_MAP:
        if marker in amount_str:
            return score
    return 1

# ── Raw data fetching ─────────────────────────────────────────

def _fetch_senate_watcher():
    for url in _SOURCES:
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "trading-bot/1.0"})
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"Politician tracker: {len(data)} Senate records from {url.split('/')[2]}")
                return data, url.split('/')[2]
        except Exception as e:
            print(f"  Senate watcher {url.split('/')[2]}: {e}")
    return [], None


def _fetch_fmp():
    if not FMP_KEY:
        return []
    records = []
    for url, chamber in [(FMP_SENATE, "senate"), (FMP_HOUSE, "house")]:
        try:
            resp = requests.get(url, params={"apikey": FMP_KEY}, timeout=15)
            resp.raise_for_status()
            for rec in (resp.json() or []):
                name = rec.get("senator") or rec.get("representative") or ""
                records.append({
                    "first_name":    name.split()[0] if name else "",
                    "last_name":     " ".join(name.split()[1:]) if name else "",
                    "date_received": rec.get("disclosureDate") or "",
                    "transactions": [{
                        "transaction_date": rec.get("transactionDate") or "",
                        "ticker":           (rec.get("ticker") or "").upper().strip(),
                        "asset_description": rec.get("assetDescription") or "",
                        "type":             rec.get("type") or "",
                        "amount":           rec.get("amount") or "",
                        "owner":            rec.get("owner") or "Self",
                    }],
                    "chamber": chamber,
                })
        except Exception as e:
            print(f"  FMP {chamber}: {e}")
    if records:
        print(f"  FMP: {len(records)} records (Senate + House)")
    return records


def _load_raw():
    """Fetch all raw trade records, updating the 6-hour cache."""
    with _lock:
        if _raw_cache["data"] is not None and time.time() - _raw_cache["ts"] < CACHE_TTL:
            return _raw_cache["data"]

    senate_records, source = _fetch_senate_watcher()
    fmp_records = _fetch_fmp()
    combined = senate_records + fmp_records

    with _lock:
        _raw_cache["data"]   = combined
        _raw_cache["ts"]     = time.time()
        _raw_cache["source"] = source
    return combined

# ── Scoring ───────────────────────────────────────────────────

def get_ticker_scores(days=90):
    """
    Return {ticker: score} for all tickers with recent politician purchases.
    Score 0–25: reflects recency, amount, and number of unique politicians buying.
    Results cache for 6 hours.
    """
    with _lock:
        if _score_cache["data"] is not None and time.time() - _score_cache["ts"] < CACHE_TTL:
            return _score_cache["data"]

    records = _load_raw()
    cutoff  = datetime.now(pytz.utc) - timedelta(days=days)
    raw_scores   = {}   # ticker → float
    politician_names = {}  # ticker → set of names (diversity bonus)

    for record in records:
        first = record.get("first_name", "")
        last  = record.get("last_name", "")
        name  = f"{first} {last}".strip() or "unknown"
        date_received = record.get("date_received", "")

        for txn in (record.get("transactions") or []):
            try:
                ticker = (txn.get("ticker") or "").upper().strip()
                if not ticker or ticker in ("--", "N/A", "", "NONE"):
                    continue
                # Only buys/purchases
                t_type = (txn.get("type") or "").lower()
                if "purchase" not in t_type and "buy" not in t_type:
                    continue
                # Skip non-equity assets
                asset = (txn.get("asset_description") or "").lower()
                if any(skip in asset for skip in
                       ("bond", "treasury", "fund", "etf", "crypto",
                        "real estate", "note", "option")):
                    continue

                # Parse transaction date (prefer txn date, fall back to disclosure)
                date_raw = (txn.get("transaction_date") or date_received or "")[:10]
                if not date_raw or len(date_raw) < 10:
                    continue
                try:
                    tx_date = datetime.strptime(date_raw, "%Y-%m-%d").replace(tzinfo=pytz.utc)
                except ValueError:
                    continue
                if tx_date < cutoff:
                    continue

                # Recency decay: more recent = higher weight
                days_ago = max(0, (datetime.now(pytz.utc) - tx_date).days)
                if   days_ago <= 7:  recency = 1.0
                elif days_ago <= 30: recency = 0.7
                elif days_ago <= 60: recency = 0.45
                else:                recency = 0.25

                pts = _parse_amount(txn.get("amount", "")) * recency
                raw_scores[ticker] = raw_scores.get(ticker, 0) + pts
                politician_names.setdefault(ticker, set()).add(name)
            except Exception:
                continue

    # Build final scores: add diversity bonus (multiple politicians = stronger signal)
    result = {}
    for ticker, raw in raw_scores.items():
        n_pols = len(politician_names.get(ticker, set()))
        score  = raw + n_pols * 2          # 2 bonus points per unique politician
        result[ticker] = round(min(25, score), 1)

    with _lock:
        _score_cache["data"] = result
        _score_cache["ts"]   = time.time()

    if result:
        top = sorted(result.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"Politician tracker top tickers: {top}")

    return result


def get_politician_tickers(days=60, min_score=3):
    """
    Return a list of tickers with meaningful recent congressional buy activity,
    ordered by score descending. Used to expand the scan universe so these
    tickers are always evaluated even if not in the seed list.
    """
    scores = get_ticker_scores(days=days)
    tickers = [(t, s) for t, s in scores.items() if s >= min_score]
    tickers.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in tickers[:20]]


def get_status_summary():
    """Return a brief status dict for dashboard/logging use."""
    with _lock:
        cached = _raw_cache["data"]
        ts     = _raw_cache["ts"]
        src    = _raw_cache["source"]
    scores = _score_cache.get("data") or {}
    return {
        "records_loaded":    len(cached) if cached else 0,
        "source":            src,
        "tickers_tracked":   len(scores),
        "top_tickers":       sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5],
        "cache_age_mins":    round((time.time() - ts) / 60, 1) if ts else None,
    }
