"""
catalyst_tracker.py — News catalyst detection for gap-and-go strategy

Identifies stocks with fundamental drivers behind their price move:
  - SEC EDGAR 8-K real-time RSS (material event filings, no key needed)
  - EDGAR company_tickers.json (CIK → ticker mapping, refreshed daily)
  - Yahoo Finance RSS per ticker (recent headlines, no key needed)

A stock gapping up on news is far more likely to continue than one gapping
on noise. This module scores the quality of the catalyst and is used to
filter gap-and-go candidates to those with real fundamental backing.

Exports:
  get_catalyst_score(ticker)  → {score, headline, source, kw, age_hours}
  get_catalyst_tickers(min_score) → [ticker, ...]
  get_edgar_scores()          → {ticker: score, ...}  (EDGAR-only, fast)
  get_status_summary()        → str
"""

import re, json, time, threading
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET
import urllib.request
import urllib.error

# ── Keyword scoring ───────────────────────────────────────────────────────────

# Score by presence of the LONGEST matching phrase first so shorter substrings
# don't shadow more specific keywords.
_POS = sorted({
    # FDA / biotech — highest impact for small caps
    "fda approved": 20, "fda approval": 20, "approved by fda": 20,
    "nda approved": 18, "bla approved": 18, "510k cleared": 15,
    "breakthrough therapy designation": 14, "fast track designation": 12,
    "pdufa date": 12, "phase 3 positive": 14, "phase 2 positive": 10,
    "positive phase": 10, "clinical trial results": 8,
    # M&A / corporate
    "definitive merger agreement": 20, "merger agreement": 18,
    "definitive agreement": 16, "tender offer": 16,
    "acquisition agreement": 16, "acquired by": 15, "acquisition": 14,
    "buyout offer": 16, "buyout": 15, "going private": 14, "merger": 13,
    # Contracts / partnerships
    "department of defense contract": 14, "government contract": 12,
    "contract awarded": 12, "awarded contract": 12,
    "strategic partnership": 10, "collaboration agreement": 10,
    "license agreement": 9, "partnership agreement": 8, "partnership": 7,
    # Earnings / guidance (medium impact)
    "earnings beat": 10, "beats estimates": 10, "beat expectations": 10,
    "record revenue": 8, "record earnings": 8, "record quarter": 8,
    "raised guidance": 10, "increases guidance": 10, "raises outlook": 10,
    # Patents / IP
    "patent granted": 8, "patent approved": 8, "patent awarded": 8,
    # Other positive
    "positive results": 6, "breakthrough": 8, "milestone achieved": 7,
    "short squeeze": 10,
}.items(), key=lambda x: -len(x[0]))

_NEG = sorted({
    "bankruptcy protection": -22, "chapter 11 protection": -22,
    "chapter 11": -20, "chapter 7": -20, "bankruptcy": -18,
    "nasdaq delisting notice": -20, "nyse delisting notice": -20,
    "delisting notice": -18, "delisted": -16,
    "sec fraud charges": -18, "sec investigation": -15, "sec charges": -15,
    "securities fraud": -15, "accounting fraud": -15, "fraud": -12,
    "class action lawsuit": -12, "class action": -10,
    "complete response letter": -15, "crl issued": -14,
    "fda rejected": -18, "fda rejection": -18, "fda refusal": -16,
    "failed phase 3": -14, "phase 3 failure": -14, "trial failed": -12,
    "missed estimates": -10, "earnings miss": -10, "misses estimates": -10,
    "lowered guidance": -10, "reduces guidance": -10, "cuts forecast": -10,
    "disappointing results": -8, "investigation": -6, "lawsuit": -5,
}.items(), key=lambda x: -len(x[0]))


def _score_text(text):
    """Keyword-scan a headline. Returns (score, best_matched_keyword)."""
    t = text.lower()
    best_score, best_kw = 0, ""
    for kw, pts in _POS:
        if kw in t and pts > best_score:
            best_score, best_kw = pts, kw
    for kw, pts in _NEG:
        if kw in t and abs(pts) > abs(best_score):
            best_score, best_kw = pts, kw
    return best_score, best_kw


# ── Caches ────────────────────────────────────────────────────────────────────

_catalyst_cache = {}   # ticker → {score, headline, source, kw, age_hours, fetched_at}
_catalyst_lock  = threading.Lock()
_CATALYST_TTL   = 3600   # 1 hour per ticker

_cik_map        = {}   # str(CIK).zfill(10) → ticker
_cik_map_ts     = 0.0
_CIK_MAP_TTL    = 86400  # refresh daily

_edgar_cache    = {}   # ticker → {score, headline, kw, fetched_at}
_edgar_ts       = 0.0
_EDGAR_TTL      = 1800   # 30 minutes

_edgar_lock     = threading.Lock()

# ── Network helper ────────────────────────────────────────────────────────────

_HEADERS = {"User-Agent": "TradingBot research-bot@example.com"}

def _get(url, timeout=10):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_8K_URL  = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)


def _load_cik_map():
    """Load CIK → ticker mapping from SEC. Cached daily."""
    global _cik_map_ts
    now_ts = time.time()
    if now_ts - _cik_map_ts < _CIK_MAP_TTL and _cik_map:
        return
    try:
        data = json.loads(_get(_CIK_URL, timeout=15))
        mapping = {}
        for entry in data.values():
            cik    = str(entry.get("cik_str", "")).zfill(10)
            ticker = entry.get("ticker", "").upper()
            if ticker:
                mapping[cik] = ticker
        with _edgar_lock:
            _cik_map.clear()
            _cik_map.update(mapping)
            _cik_map_ts = now_ts
        print(f"Catalyst: EDGAR CIK map loaded — {len(mapping)} tickers")
    except Exception as e:
        print(f"Catalyst: EDGAR CIK map error: {e}")


def _fetch_edgar_8k():
    """
    Fetch the 40 most recent 8-K filings from EDGAR RSS.
    Returns dict: ticker → {score, headline, kw, fetched_at}.
    Cached for _EDGAR_TTL seconds (30 min).
    """
    global _edgar_ts
    now_ts = time.time()
    if now_ts - _edgar_ts < _EDGAR_TTL:
        return _edgar_cache

    _load_cik_map()
    results = {}

    try:
        xml  = _get(_8K_URL)
        root = ET.fromstring(xml)
        ns   = {"a": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("a:entry", ns):
            title   = entry.findtext("a:title",   "", ns).strip()
            summary = entry.findtext("a:summary", "", ns).strip()
            link_el = entry.find("a:link", ns)
            href    = link_el.get("href", "") if link_el is not None else ""

            m = re.search(r"CIK=(\d+)", href, re.I)
            if not m:
                continue
            cik    = m.group(1).zfill(10)
            ticker = _cik_map.get(cik)
            if not ticker:
                continue

            text  = f"{title} {summary}"
            score, kw = _score_text(text)
            tl = text.lower()

            # Boost based on high-value 8-K items even if keywords didn't match
            if any(x in tl for x in ["2.01", "completion of acquisition"]):
                score = max(score, 15)
            elif any(x in tl for x in ["1.01", "material definitive agreement"]):
                score = max(score, 10)
            elif any(x in tl for x in ["2.02", "results of operations"]):
                score = max(score, 6)
            elif any(x in tl for x in ["8.01", "7.01"]):
                score = max(score, 4)   # general press release filed

            if score != 0:
                results[ticker] = {
                    "score":      score,
                    "headline":   title[:100],
                    "source":     "edgar_8k",
                    "kw":         kw,
                    "fetched_at": now_ts,
                }

    except Exception as e:
        print(f"Catalyst: EDGAR 8-K fetch error: {e}")

    with _edgar_lock:
        _edgar_cache.clear()
        _edgar_cache.update(results)
        _edgar_ts = now_ts

    if results:
        print(f"Catalyst: EDGAR 8-K — {len(results)} catalyst tickers")
    return results


# ── Yahoo Finance RSS ─────────────────────────────────────────────────────────

_YAHOO_RSS = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={ticker}&region=US&lang=en-US"
)


def _fetch_yahoo_news(ticker, max_age_hours=48):
    """
    Fetch recent Yahoo Finance headlines for a specific ticker.
    Returns list of {title, age_h} sorted newest-first.
    """
    try:
        xml  = _get(_YAHOO_RSS.format(ticker=ticker))
        root = ET.fromstring(xml)
        items = []
        for item in root.iter("item"):
            title   = item.findtext("title", "").strip()
            pub_str = item.findtext("pubDate", "")
            try:
                pub_dt = parsedate_to_datetime(pub_str)
                age_h  = (datetime.now(pub_dt.tzinfo) - pub_dt).total_seconds() / 3600
            except Exception:
                age_h = 0
            if age_h <= max_age_hours and title:
                items.append({"title": title, "age_h": round(age_h, 1)})
        return items
    except Exception as e:
        # RSS can 404 for tickers Yahoo doesn't cover — that's normal
        if "404" not in str(e) and "HTTP Error" not in str(e):
            print(f"Catalyst: Yahoo news {ticker}: {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_catalyst_score(ticker):
    """
    Returns the best catalyst found for ticker in the last 48 hours.
    Checks EDGAR 8-K RSS (batch, fast) then Yahoo Finance (per-ticker).
    Result is cached for _CATALYST_TTL (1 hour).

    Returns dict: {score, headline, source, kw, age_hours, fetched_at}
    score > 0: bullish catalyst.  score < 0: negative event.
    """
    now_ts = time.time()
    cached = _catalyst_cache.get(ticker)
    if cached and now_ts - cached.get("fetched_at", 0) < _CATALYST_TTL:
        return cached

    with _catalyst_lock:
        cached = _catalyst_cache.get(ticker)
        if cached and now_ts - cached.get("fetched_at", 0) < _CATALYST_TTL:
            return cached

        best = {"score": 0, "headline": "", "source": "none",
                "kw": "", "age_hours": 999, "fetched_at": now_ts}

        # EDGAR (batch-cached, fast)
        edgar = _fetch_edgar_8k()
        if ticker in edgar:
            e = edgar[ticker]
            if abs(e["score"]) > abs(best["score"]):
                best = {**e, "age_hours": 0, "fetched_at": now_ts}

        # Yahoo Finance (per-ticker HTTP call)
        for item in _fetch_yahoo_news(ticker, max_age_hours=48):
            score, kw = _score_text(item["title"])
            if abs(score) > abs(best["score"]):
                best = {
                    "score":      score,
                    "headline":   item["title"][:120],
                    "source":     "yahoo",
                    "kw":         kw,
                    "age_hours":  item["age_h"],
                    "fetched_at": now_ts,
                }

        _catalyst_cache[ticker] = best
        return best


def get_catalyst_tickers(min_score=5):
    """Return tickers from recent EDGAR 8-Ks with score >= min_score."""
    return [t for t, d in _fetch_edgar_8k().items() if d.get("score", 0) >= min_score]


def get_edgar_scores():
    """
    Return {ticker: score} for all tickers with recent 8-K filings.
    Backed by the 30-min EDGAR cache — no per-ticker HTTP calls.
    """
    return {t: d.get("score", 0) for t, d in _fetch_edgar_8k().items()}


def get_status_summary():
    return (
        f"Catalyst: {len(_catalyst_cache)} tickers cached | "
        f"EDGAR 8-K: {len(_edgar_cache)} recent filings | "
        f"CIK map: {len(_cik_map)} entries"
    )
