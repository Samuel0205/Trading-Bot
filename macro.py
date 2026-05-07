"""
macro.py — Earnings calendar + macro event awareness

Sources (all free):
  - Alpaca news API: scans for earnings keywords
  - SEC EDGAR: checks for 8-K filings (earnings releases)
  - Federal Reserve calendar: FOMC meeting dates (hardcoded + updateable)
  - BLS economic calendar: CPI/jobs dates (hardcoded + updateable)

FIX vs original:
  - JOBS_DATES_2025 was defined but never passed to seed_macro_calendar() — now included.
  - check_sec_8k() is now called inside check_earnings_risk() as a secondary signal.
"""

import requests, time, json
from datetime import datetime, timedelta, date
import pytz
from database import save_macro_event, is_macro_blackout, save_alert, get_upcoming_macro_events

NY = pytz.timezone("America/New_York")

# ── FOMC dates ────────────────────────────────────────────────

FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07",
    "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-17",
]
FOMC_DATES_2026 = [
    "2026-01-28","2026-03-18","2026-05-06",
    "2026-06-17","2026-07-29","2026-09-16",
    "2026-10-28","2026-12-16",
]

# ── CPI dates ─────────────────────────────────────────────────

CPI_DATES_2025 = [
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10",
    "2025-05-13","2025-06-11","2025-07-11","2025-08-12",
    "2025-09-10","2025-10-15","2025-11-12","2025-12-10",
]
CPI_DATES_2026 = [
    "2026-01-14","2026-02-11","2026-03-11","2026-04-09",
    "2026-05-13","2026-06-10","2026-07-14","2026-08-12",
    "2026-09-11","2026-10-14","2026-11-13","2026-12-10",
]

# ── Jobs report dates ─────────────────────────────────────────

JOBS_DATES_2025 = [                          # FIX: was defined but never seeded
    "2025-01-10","2025-02-07","2025-03-07","2025-04-04",
    "2025-05-02","2025-06-06","2025-07-03","2025-08-01",
    "2025-09-05","2025-10-03","2025-11-07","2025-12-05",
]
JOBS_DATES_2026 = [
    "2026-01-09","2026-02-06","2026-03-06","2026-04-03",
    "2026-05-08","2026-06-05","2026-07-02","2026-07-31",
    "2026-09-04","2026-10-02","2026-11-06","2026-12-04",
]

# ── Sector ETFs ───────────────────────────────────────────────

SECTOR_ETFS = {
    "Technology":    "XLK",
    "Energy":        "XLE",
    "Finance":       "XLF",
    "Healthcare":    "XLV",
    "Consumer":      "XLY",
    "Industrials":   "XLI",
    "Materials":     "XLB",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Communication": "XLC",
}

# ── Initialize macro calendar ─────────────────────────────────

def seed_macro_calendar():
    """
    Populate DB with all known macro events.
    FIX: JOBS_DATES_2025 is now included (was missing in original).
    """
    print("Seeding macro calendar...")
    all_events = (
        [(d, "FOMC Meeting",  "high", "federal_reserve") for d in FOMC_DATES_2025 + FOMC_DATES_2026] +
        [(d, "CPI Release",   "high", "BLS")             for d in CPI_DATES_2025  + CPI_DATES_2026]  +
        [(d, "Jobs Report",   "high", "BLS")             for d in JOBS_DATES_2025 + JOBS_DATES_2026]  # FIX
    )
    seeded = 0
    for event_date, name, impact, source in all_events:
        try:
            save_macro_event(event_date, name, impact, source)
            seeded += 1
        except:
            pass
    print(f"Macro calendar: {seeded}/{len(all_events)} events seeded")

# ── SEC 8-K check ─────────────────────────────────────────────

def check_sec_8k(ticker):
    """
    Checks SEC EDGAR for recent 8-K filings.
    8-K Item 2.02 = Results of Operations (earnings release).
    Free, no API key needed.
    FIX: now called from check_earnings_risk() instead of being dead code.
    """
    try:
        start_dt = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        end_dt   = date.today().strftime("%Y-%m-%d")
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}&forms=8-K"
        )
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "trading-bot contact@example.com"})
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            # Item 2.02 = Results of Operations
            earnings_8k = [h for h in hits
                           if "2.02" in str(h.get("_source", {}).get("items", ""))]
            if earnings_8k:
                return True, f"SEC 8-K Item 2.02 detected ({len(earnings_8k)} filing(s))"
        return False, ""
    except Exception as e:
        print(f"  check_sec_8k error {ticker}: {e}")
        return False, ""

# ── Earnings detection ────────────────────────────────────────

EARNINGS_KEYWORDS = [
    "earnings", "eps", "quarterly results", "revenue", "guidance",
    "q1 ", "q2 ", "q3 ", "q4 ", "fiscal quarter", "beat", "miss",
    "profit", "income report", "quarterly report"
]

def check_earnings_risk(api, ticker, days_window=3):
    """
    Checks if ticker has earnings coming in next N days or just reported.
    Now also checks SEC EDGAR 8-K filings as a secondary signal.
    Returns: (risk_level, score_adj, detail)
    """
    try:
        now   = datetime.now(pytz.utc)
        start = now - timedelta(days=days_window)
        end   = now + timedelta(days=days_window)
        news  = api.get_news(
            ticker,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=20
        )

        earnings_hits = []
        if news:
            for n in news:
                headline = n.headline.lower()
                matches  = [kw for kw in EARNINGS_KEYWORDS if kw in headline]
                if matches:
                    earnings_hits.append({
                        "headline": n.headline,
                        "keywords": matches,
                        "date":     str(n.created_at)[:10]
                    })

        # SEC 8-K check — secondary earnings signal
        sec_hit, sec_detail = check_sec_8k(ticker)
        if sec_hit:
            earnings_hits.append({"headline": sec_detail, "keywords": ["8-K"], "date": str(date.today())})

        if len(earnings_hits) >= 3:
            save_alert("WARN", f"{ticker} earnings risk HIGH ({len(earnings_hits)} signals) — skipping", ticker)
            return "high", -25, f"{len(earnings_hits)} earnings signals (news + SEC)"
        elif len(earnings_hits) >= 1:
            return "medium", -10, f"{len(earnings_hits)} earnings signal"
        else:
            return "low", 0, "no earnings signals"

    except Exception as e:
        print(f"earnings_risk error {ticker}: {e}")
        return "unknown", 0, "check failed"

# ── Macro blackout check ──────────────────────────────────────

def is_macro_blackout_today():
    """Returns (True, event_name) if today is a high-impact macro event day."""
    today = datetime.now(NY).strftime("%Y-%m-%d")
    return is_macro_blackout(today)

def get_next_macro_event():
    """Returns the next upcoming macro event for display."""
    events = get_upcoming_macro_events(days=14)
    return events[0] if events else None

# ── Sector rotation ───────────────────────────────────────────

def get_sector_momentum(api):
    """
    Fetches 5-day performance of each sector ETF.
    Returns ranked dict showing which sectors are hot/cold.
    """
    results = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=8)
            bars  = api.get_bars(etf, "1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=6, feed="iex").df

            if bars is None or bars.empty or len(bars) < 2:
                continue

            if hasattr(bars.index, 'levels'):
                if etf in bars.index.get_level_values(0):
                    bars = bars.loc[etf]
                else:
                    continue

            closes   = list(bars["close"])
            five_day = closes[-min(5, len(closes))]
            latest   = closes[-1]
            chg_pct  = round((latest - five_day) / five_day * 100, 2) if five_day > 0 else 0
            volumes  = list(bars["volume"])
            avg_vol  = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
            rvol     = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

            results[sector] = {
                "etf":      etf,
                "price":    round(latest, 2),
                "chg_5d":   chg_pct,
                "rvol":     round(rvol, 2),
                "momentum": "hot"  if chg_pct >  2 else
                            "cold" if chg_pct < -2 else "neutral",
            }
            time.sleep(0.2)
        except Exception as e:
            print(f"  Sector {etf} error: {e}")
            continue

    return dict(sorted(results.items(), key=lambda x: x[1]["chg_5d"], reverse=True))

def get_hot_sectors(api, top_n=3):
    """Returns list of hot sector names."""
    momentum = get_sector_momentum(api)
    hot = [s for s, d in momentum.items() if d["momentum"] == "hot"]
    return hot[:top_n]

# ── Unusual volume scanner ────────────────────────────────────

def scan_unusual_volume(api, universe, account_size=20):
    """
    Free-tier proxy for options flow.
    Unusual volume often precedes big moves.
    Returns top candidates sorted by anomaly score.
    """
    results = []
    floor   = 0.50
    ceiling = min(account_size * 0.45, 10.0)

    for ticker in universe:
        try:
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=10)
            bars  = api.get_bars(ticker, "1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10, feed="iex").df

            if bars is None or bars.empty or len(bars) < 5:
                continue

            if hasattr(bars.index, 'levels'):
                if ticker in bars.index.get_level_values(0):
                    bars = bars.loc[ticker]
                else:
                    continue

            closes  = list(bars["close"])
            volumes = list(bars["volume"])
            price   = closes[-1]

            if not (floor <= price <= ceiling):
                continue

            avg_vol   = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
            today_vol = volumes[-1]
            rvol      = today_vol / avg_vol if avg_vol > 0 else 1
            price_chg = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
            anomaly   = rvol * abs(price_chg)

            if rvol >= 2.0:
                results.append({
                    "ticker":        ticker,
                    "price":         round(price, 2),
                    "rvol":          round(rvol, 2),
                    "price_chg_pct": round(price_chg, 2),
                    "anomaly_score": round(anomaly, 2),
                    "direction":     "up" if price_chg > 0 else "down",
                })
            time.sleep(0.15)
        except:
            continue

    results.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return results[:10]

# ── Full macro status ─────────────────────────────────────────

def get_macro_status(api):
    """Returns comprehensive macro status for dashboard display."""
    blackout, event_name = is_macro_blackout_today()
    next_event           = get_next_macro_event()

    status = {
        "blackout":       blackout,
        "blackout_event": event_name,
        "next_event":     next_event,
        "sectors":        {},
        "timestamp":      datetime.now(NY).strftime("%I:%M %p ET"),
    }

    try:
        status["sectors"] = get_sector_momentum(api)
    except Exception as e:
        print(f"Sector momentum error: {e}")

    return status
