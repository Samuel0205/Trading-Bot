"""
scanner.py — Stock scanner + universe builder

Builds a dynamic universe of affordable, liquid stocks each scan cycle.
Uses IEX feed (free Alpaca tier). No circular imports.

Exports required by app.py:
  - SEED_UNIVERSE  (list of ticker strings)
  - run_full_scan(api)  → dict with today/yesterday/meta

Account sizing parameters are passed INTO run_full_scan rather than
imported from app.py, which would cause a circular import.
"""

import time, os
from datetime import datetime, timedelta
import pytz
from finbert_client import finbert_score, keyword_score

NY          = pytz.timezone("America/New_York")
MAX_ACCOUNT = float(os.environ.get("MAX_ACCOUNT", "30.00"))

def get_headlines(api, ticker, days_back=1):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days_back)
        news  = api.get_news(ticker,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=10)
        return [n.headline for n in news] if news else []
    except:
        return []

# ── Seed universe ─────────────────────────────────────────────

SEED_UNIVERSE = [
    "SIRI","TELL","CLOV","NKLA","MVIS","SOFI","HOOD","NIO","MARA","RIOT",
    "PLTR","RIVN","LCID","HBAN","KEY","VALE","ITUB","PBR","GOLD",
    "KGC","HL","CDE","AG","EGO","BTG","NGD","PAAS","SILV",
    "AFRM","OPEN","DKNG","CHPT","WKHS","HYLN","SNDL","ACB",
    "CGC","TLRY","CRON","CWEB","SPCE","MAXN","ARRY",
    "STEM","NOVA","SHLS","IDEX","AMC","BB","NOK",
    "OCGN","TIGR","XPEV","LI","JOBY","PTON","CLSK","BITF",
    "F","BAC","AAL","CCL","SNAP","KOSS","RIG","CLF",
]
SEED_UNIVERSE = list(dict.fromkeys(SEED_UNIVERSE))  # deduplicate, preserve order

# ── Account sizing (no import from app — avoids circular import) ──

def _get_account_size(api):
    """
    Local account size — does NOT import from app.py (avoids circular import).
    FIX: Returns real equity without hard cap, matching app.py behavior.
    MAX_ACCOUNT is only the fallback when the API call fails.
    """
    try:
        return float(api.get_account().equity)
    except:
        return MAX_ACCOUNT

def _price_floor(account_size):
    if account_size > 100: return 2.00
    if account_size > 20:  return 1.00
    return 0.50

def _price_ceiling(account_size):
    if account_size > 50_000: return min(account_size * 0.45, 50.00)
    if account_size > 10_000: return min(account_size * 0.45, 25.00)
    return max(min(account_size * 0.45, 10.00), 0.60)

def _min_volume(account_size):
    # Calibrated for IEX exchange feed (2-3% of consolidated NASDAQ/NYSE volume).
    # The old 1M threshold meant only stocks trading 50M+ real shares/day passed,
    # which biased the universe toward perpetually-downtrending meme stocks.
    # Matches get_min_volume in bot.py.
    if account_size > 5000: return 2_000
    if account_size > 1000: return 1_000
    if account_size > 200:  return 500
    return 200

# ── Bar fetching helpers ──────────────────────────────────────

def safe_get_bars(api, symbols, timeframe="1Day", days_back=5, limit=5):
    """Always uses feed=iex — required for free Alpaca accounts."""
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days_back + 2)
        bars  = api.get_bars(
            symbols, timeframe,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=limit, feed="iex"
        ).df
        return bars if not bars.empty else None
    except Exception as e:
        print(f"  get_bars error: {e}")
        return None

def extract_sym(bars_df, sym):
    """Safely extract a single symbol from a potentially MultiIndex DataFrame."""
    try:
        if bars_df is None:
            return None
        if hasattr(bars_df.index, 'levels'):
            if sym not in bars_df.index.get_level_values(0):
                return None
            return bars_df.loc[sym]
        return bars_df
    except:
        return None

# ── Universe builder ──────────────────────────────────────────

def build_universe(api, max_price, min_price=0.50, min_volume=100_000):
    """
    Checks seed list first (fast). Expands with asset list if < 15 found.
    Returns list of {symbol, price, volume} dicts.
    """
    print(f"  Building universe: ${min_price}–${max_price:.2f}, vol>{min_volume:,}")
    universe   = []
    seen       = set()
    batch_size = 50

    # Step 1 — seed universe (fast path)
    for i in range(0, len(SEED_UNIVERSE), batch_size):
        batch = SEED_UNIVERSE[i:i+batch_size]
        bars  = safe_get_bars(api, batch)
        if bars is None:
            continue
        for sym in batch:
            if sym in seen:
                continue
            sym_bars = extract_sym(bars, sym)
            if sym_bars is None or len(sym_bars) < 2:
                continue
            try:
                price   = float(sym_bars.iloc[-1]["close"])
                avg_vol = float(sym_bars["volume"].mean())
                if min_price <= price <= max_price and avg_vol >= min_volume:
                    universe.append({"symbol": sym, "price": round(price, 2), "volume": int(avg_vol)})
                    seen.add(sym)
            except:
                continue
        time.sleep(0.3)

    print(f"  Seed scan: {len(universe)} stocks")

    # Step 2 — expand if needed
    if len(universe) < 15:
        print("  Expanding with asset list (up to 300)...")
        try:
            assets    = api.list_assets(status="active", asset_class="us_equity")
            tradeable = [
                a.symbol for a in assets
                if a.tradable
                and a.exchange in ("NYSE", "NASDAQ", "ARCA")
                and not a.symbol.endswith(("W", "R", "P", "Q"))
                and len(a.symbol) <= 5
                and a.symbol not in seen
            ][:300]

            for i in range(0, len(tradeable), batch_size):
                batch = tradeable[i:i+batch_size]
                bars  = safe_get_bars(api, batch)
                if bars is None:
                    continue
                for sym in batch:
                    if sym in seen:
                        continue
                    sym_bars = extract_sym(bars, sym)
                    if sym_bars is None or len(sym_bars) < 2:
                        continue
                    try:
                        price   = float(sym_bars.iloc[-1]["close"])
                        avg_vol = float(sym_bars["volume"].mean())
                        if min_price <= price <= max_price and avg_vol >= min_volume:
                            universe.append({"symbol": sym, "price": round(price, 2), "volume": int(avg_vol)})
                            seen.add(sym)
                    except:
                        continue
                time.sleep(0.3)
        except Exception as e:
            print(f"  Asset expand error: {e}")

    print(f"  Universe built: {len(universe)} stocks total")
    return universe

# ── Price data fetcher ────────────────────────────────────────

def get_price_data(api, ticker):
    try:
        bars = safe_get_bars(api, ticker, days_back=10, limit=10)
        if bars is None:
            return None
        if hasattr(bars.index, 'levels'):
            bars = extract_sym(bars, ticker)
            if bars is None:
                return None
        if len(bars) < 2:
            return None

        latest     = bars.iloc[-1]
        prev       = bars.iloc[-2]
        price      = round(float(latest["close"]), 2)
        prev_close = round(float(prev["close"]), 2)
        if prev_close == 0:
            return None
        change_pct = round((price - prev_close) / prev_close * 100, 2)
        avg_vol    = float(bars["volume"].mean())
        vol_ratio  = round(float(latest["volume"]) / avg_vol, 2) if avg_vol > 0 else 1.0
        five_day   = float(bars.iloc[max(0, len(bars)-5)]["close"])
        rs_5d      = round((price - five_day) / five_day * 100, 2) if five_day > 0 else 0.0
        avg_range  = float(((bars["high"] - bars["low"]) / bars["close"].replace(0, 1)).mean() * 100)

        return {
            "price":      price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "vol_ratio":  vol_ratio,
            "rs_5d":      rs_5d,
            "avg_range":  round(avg_range, 2),
        }
    except Exception as e:
        print(f"  price_data error {ticker}: {e}")
        return None

# ── Composite scoring ─────────────────────────────────────────

def composite_score(pd, sentiment, news_count, account_size,
                    politician_score=0, social_score=0, short_score=0, insider_score=0,
                    catalyst_score=0):
    s  = abs(pd["change_pct"]) * 2.5
    s += abs(pd["rs_5d"])      * 1.5
    vr = pd["vol_ratio"]
    if   vr > 5.0: s += 25
    elif vr > 3.0: s += 18
    elif vr > 2.0: s += 10
    elif vr > 1.5: s +=  5
    elif vr > 1.2: s +=  2
    s += sentiment * 8
    s += min(news_count * 2, 12)
    if account_size < 100:
        s += pd["avg_range"] * 2
    elif account_size < 500:
        s += pd["avg_range"] * 1
    if account_size < 50:
        if   pd["price"] < 2: s += 10
        elif pd["price"] < 5: s +=  5
    elif account_size < 200:
        if   pd["price"] < 5: s +=  8
        elif pd["price"] < 8: s +=  3
    if politician_score > 0:
        s += politician_score * 3.5
    # Social sentiment: crowd is directly relevant for meme/momentum universe
    if social_score != 0:
        s += social_score * 1.5   # -15 to +15
    # Short squeeze potential: days-to-cover + current rvol context
    if short_score > 0:
        s += short_score * 1.2    # 0 to +18
    # Insider (Form 4) buying: slower signal but high conviction
    if insider_score > 0:
        s += insider_score * 2.5  # 0 to +50 at max, capped below
    if catalyst_score != 0:
        s += catalyst_score * 2.0  # EDGAR/news catalyst quality boost
    return round(s, 2)

def risk_grade(pd):
    score = 0
    if pd["vol_ratio"]       < 3:  score += 2
    if abs(pd["change_pct"]) < 5:  score += 2
    if abs(pd["rs_5d"])      < 10: score += 2
    if pd["avg_range"]       < 5:  score += 2
    if pd["avg_range"]       < 3:  score += 1
    return {9:"A",8:"A",7:"B",6:"B",5:"C",4:"C",3:"D"}.get(score, "F")

# ── Per-day scan ──────────────────────────────────────────────

def run_scan(api, universe, days_back=1, account_size=20):
    label    = "today" if days_back == 1 else "yesterday"
    max_price = _price_ceiling(account_size)   # consistent with bot.py passes_filters
    print(f"\n=== Scanning {len(universe)} stocks ({label}) | acct ${account_size:.2f} "
          f"max ${max_price:.2f} ===")
    results = []

    # ── Fetch all signal caches once per scan cycle ───────────────────────
    pol_scores    = {}
    insider_scores = {}
    social_data   = {}

    try:
        from politician_tracker import get_ticker_scores
        pol_scores = get_ticker_scores()
        if pol_scores:
            print(f"  Politician scores: {len(pol_scores)} tickers tracked")
    except Exception as e:
        print(f"  politician_tracker unavailable: {e}")

    try:
        from insider_tracker import get_insider_scores
        insider_scores = get_insider_scores()
        if insider_scores:
            print(f"  Insider scores: {len(insider_scores)} tickers tracked")
    except Exception as e:
        print(f"  insider_tracker unavailable: {e}")

    edgar_scores = {}
    try:
        from catalyst_tracker import get_edgar_scores
        edgar_scores = get_edgar_scores()
        if edgar_scores:
            print(f"  EDGAR catalyst scores: {len(edgar_scores)} tickers")
    except Exception as e:
        print(f"  catalyst_tracker unavailable: {e}")

    tickers_list = [s["symbol"] for s in universe]
    try:
        from social_sentiment import get_social_batch
        social_data = get_social_batch(tickers_list)
    except Exception as e:
        print(f"  social_sentiment unavailable: {e}")

    for stock in universe:
        ticker = stock["symbol"]
        try:
            pd = get_price_data(api, ticker)
            if not pd:
                continue
            # Apply same floor+ceiling as bot.py passes_filters so scan results
            # only contain stocks that will actually pass the entry gate.
            min_price = _price_floor(account_size)
            if pd["price"] <= 0 or pd["price"] > max_price or pd["price"] < min_price:
                continue
            headlines         = get_headlines(api, ticker, days_back=days_back)
            sentiment, method = finbert_score(headlines)
            pol_score         = pol_scores.get(ticker, 0)
            insider_score     = insider_scores.get(ticker, 0)
            soc               = social_data.get(ticker, {})
            social_score      = soc.get("social", 0)
            short_score       = soc.get("short", 0)
            cat_score         = edgar_scores.get(ticker, 0)
            score             = composite_score(
                pd, sentiment, len(headlines), account_size,
                politician_score=pol_score,
                social_score=social_score,
                short_score=short_score,
                insider_score=insider_score,
                catalyst_score=cat_score,
            )
            grade             = risk_grade(pd)
            results.append({
                "ticker":         ticker,
                "price":          pd["price"],
                "change_pct":     pd["change_pct"],
                "vol_ratio":      pd["vol_ratio"],
                "rs_5d":          pd["rs_5d"],
                "avg_range":      pd["avg_range"],
                "sentiment":      sentiment,
                "sent_method":    method,
                "news_count":     len(headlines),
                "score":          score,
                "grade":          grade,
                "direction":      "up" if pd["change_pct"] > 0 else "down",
                "pol_score":      pol_score,
                "social_score":   round(social_score, 2),
                "short_score":    round(short_score, 2),
                "insider_score":  insider_score,
                "squeeze":        soc.get("squeeze", False),
                "catalyst_score": cat_score,
            })
            time.sleep(0.1)
        except Exception as e:
            print(f"  Scan error {ticker}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"=== Top 5 ({label}): {[(r['ticker'], r['grade'], r['score']) for r in top5]} ===")
    return top5

# ── Full scan entry point — called by app.py ─────────────────

def run_full_scan(api):
    """
    Main entry point called by app.py scanner_loop and /scan/manual.
    Does NOT import from app.py to avoid circular imports.
    Derives account sizing independently.
    """
    account_size = _get_account_size(api)
    max_price    = _price_ceiling(account_size)
    min_price    = _price_floor(account_size)
    min_vol      = _min_volume(account_size)

    print(f"Full scan | account ${account_size:.2f} | range ${min_price}–${max_price:.2f}")

    universe = build_universe(api, max_price, min_price, min_vol)

    # Hard-include tickers flagged by congressional and insider buying signals.
    # They're added to the universe so run_scan() evaluates them and applies
    # the score boosts — they still must pass price/volume filters.
    pol_tickers = []
    try:
        from politician_tracker import get_politician_tickers
        pol_tickers = get_politician_tickers()
        existing    = {s["symbol"] for s in universe}
        added       = [t for t in pol_tickers if t not in existing]
        for t in added:
            universe.append({"symbol": t, "price": 0.0, "volume": 0})
        if added:
            print(f"  Politician-tracked tickers added to universe: {added}")
    except Exception as e:
        print(f"  politician_tracker expand error: {e}")

    try:
        from insider_tracker import get_insider_tickers
        insider_tickers = get_insider_tickers()
        existing        = {s["symbol"] for s in universe}
        added_ins       = [t for t in insider_tickers if t not in existing]
        for t in added_ins:
            universe.append({"symbol": t, "price": 0.0, "volume": 0})
        if added_ins:
            print(f"  Insider-tracked tickers added to universe: {added_ins}")
    except Exception as e:
        print(f"  insider_tracker expand error: {e}")

    try:
        from catalyst_tracker import get_catalyst_tickers
        cat_tickers = get_catalyst_tickers(min_score=5)
        existing    = {s["symbol"] for s in universe}
        added_cat   = [t for t in cat_tickers if t not in existing]
        for t in added_cat:
            universe.append({"symbol": t, "price": 0.0, "volume": 0})
        if added_cat:
            print(f"  Catalyst-tracked tickers added to universe: {added_cat[:10]}")
    except Exception as e:
        print(f"  catalyst_tracker expand error: {e}")

    if not universe:
        print("Universe empty — using fallback")
        universe = [{"symbol": s, "price": 1.0, "volume": 500_000}
                    for s in ["SIRI","TELL","CLOV","NKLA","MVIS",
                               "SOFI","HOOD","NIO","MARA","RIOT"]]

    today     = run_scan(api, universe, days_back=1, account_size=account_size)
    yesterday = run_scan(api, universe, days_back=2, account_size=account_size)

    return {
        "today":               today,
        "yesterday":           yesterday,
        "scanned_at":          datetime.now(NY).strftime("%I:%M %p ET"),
        "account_size":        round(account_size, 2),
        "price_range":         f"${min_price}–${max_price:.2f}",
        "universe_size":       len(universe),
        "politician_tickers":  pol_tickers[:10],
    }
