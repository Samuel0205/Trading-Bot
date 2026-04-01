import time, os, requests as req
from datetime import datetime, timedelta
import pytz

HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

NY          = pytz.timezone("America/New_York")
MAX_ACCOUNT = 20.00  # update as your account grows

# ── Sentiment ─────────────────────────────────────────────────

POSITIVE_WORDS = [
    "surge","soar","rally","beat","record","upgrade","buy","bullish","growth",
    "profit","revenue","partnership","launch","breakthrough","strong","exceed",
    "outperform","raise","acquire","expansion","dividend","momentum","breakout"
]
NEGATIVE_WORDS = [
    "crash","plunge","drop","miss","downgrade","sell","bearish","loss","decline",
    "lawsuit","recall","layoff","cut","weak","disappoint","probe","fraud",
    "bankruptcy","debt","warning","risk","halt","delisted","investigation"
]

def keyword_score(text):
    t = text.lower()
    return sum(1 for w in POSITIVE_WORDS if w in t) - sum(1 for w in NEGATIVE_WORDS if w in t)

def finbert_score(headlines):
    if not headlines:
        return 0, "no_news"
    if not HF_TOKEN:
        return sum(keyword_score(h) for h in headlines), "keyword"
    try:
        payload  = {"inputs": headlines[:5], "options": {"wait_for_model": True}}
        resp     = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=15)
        if resp.status_code != 200:
            return sum(keyword_score(h) for h in headlines), "keyword_fallback"
        total = 0
        for result in resp.json():
            sm     = {r["label"]: r["score"] for r in result}
            total += sm.get("positive", 0) - sm.get("negative", 0)
        return round(total, 3), "finbert"
    except Exception as e:
        print(f"  FinBERT error: {e}")
        return sum(keyword_score(h) for h in headlines), "keyword_fallback"

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

# ── Seed universe — known liquid low-cost stocks ──────────────
# These are checked first (fast). Scanner expands if needed.

SEED_UNIVERSE = [
    "SIRI","TELL","CLOV","NKLA","MVIS","SOFI","HOOD","NIO","MARA","RIOT",
    "PLTR","RIVN","LCID","BITI","HBAN","KEY","VALE","ITUB","PBR","GOLD",
    "KGC","HL","CDE","AG","EGO","BTG","NGD","PAAS","SILV","GPL",
    "AFRM","OPEN","DKNG","CHPT","WKHS","GOEV","FSR","HYLN","SNDL","ACB",
    "CGC","TLRY","CRON","OGI","VFF","HEXO","CWEB","SPCE","MAXN","ARRY",
    "STEM","NOVA","SHLS","SUNW","IDEX","GNUS","EXPR","AMC","BB","NOK",
    "XELA","CLOV","MVIS","TELL","SIRI","WISH","OCGN","TIGR","XPEV","LI"
]

# ── Helpers ───────────────────────────────────────────────────

def safe_get_bars(api, symbols, timeframe="1Day", days_back=5, limit=5):
    """
    Wrapper around api.get_bars that always uses feed=iex.
    Returns a DataFrame or None on failure.
    Always uses IEX feed — required for free Alpaca accounts.
    """
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days_back + 2)  # buffer for weekends
        bars  = api.get_bars(
            symbols,
            timeframe,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=limit,
            feed="iex"
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
            lvl0 = bars_df.index.get_level_values(0)
            if sym not in lvl0:
                return None
            return bars_df.loc[sym]
        return bars_df
    except:
        return None

# ── Universe builder ──────────────────────────────────────────

def build_universe(api, max_price, min_price=0.50, min_volume=100_000):
    """
    Step 1: Check seed list (fast, ~2 batches).
    Step 2: Expand with asset list if fewer than 20 found.
    All bar requests use feed=iex.
    """
    print(f"  Building universe: ${min_price}–${max_price:.2f}, vol>{min_volume:,}")
    universe   = []
    seen       = set()
    batch_size = 50

    # Step 1 — seed universe
    seeds = list(dict.fromkeys(SEED_UNIVERSE))  # deduplicated, order preserved
    for i in range(0, len(seeds), batch_size):
        batch = seeds[i:i+batch_size]
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
                    universe.append({"symbol": sym, "price": round(price,2), "volume": int(avg_vol)})
                    seen.add(sym)
            except:
                continue
        time.sleep(0.3)

    print(f"  Seed scan done: {len(universe)} stocks found")

    # Step 2 — expand if needed
    if len(universe) < 15:
        print("  Expanding with asset list (max 300)...")
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
                            universe.append({"symbol": sym, "price": round(price,2), "volume": int(avg_vol)})
                            seen.add(sym)
                    except:
                        continue
                time.sleep(0.3)
        except Exception as e:
            print(f"  Asset expand error: {e}")

    print(f"  Universe built: {len(universe)} stocks total")
    return universe

# ── Price data ────────────────────────────────────────────────

def get_price_data(api, ticker):
    """Fetch OHLCV data for a single ticker using IEX feed."""
    try:
        bars = safe_get_bars(api, ticker, days_back=10, limit=10)
        if bars is None:
            return None

        # Handle MultiIndex (single ticker may or may not be wrapped)
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

# ── Scoring ───────────────────────────────────────────────────

def composite_score(pd, sentiment, news_count, account_size):
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
    return round(s, 2)

def risk_grade(pd):
    score = 0
    if pd["vol_ratio"]         < 3:  score += 2
    if abs(pd["change_pct"])   < 5:  score += 2
    if abs(pd["rs_5d"])        < 10: score += 2
    if pd["avg_range"]         < 5:  score += 2
    if pd["avg_range"]         < 3:  score += 1
    return {9:"A",8:"A",7:"B",6:"B",5:"C",4:"C",3:"D"}.get(score, "F")

# ── Main scan ─────────────────────────────────────────────────

def run_scan(api, universe, days_back=1, account_size=20):
    label = "today" if days_back == 1 else "yesterday"
    print(f"\n=== Scanning {len(universe)} stocks ({label}) | acct ${account_size:.2f} ===")
    results = []

    for stock in universe:
        ticker = stock["symbol"]
        try:
            pd = get_price_data(api, ticker)
            if not pd:
                continue
            if pd["price"] <= 0 or pd["price"] > (account_size * 0.6):
                continue
            headlines         = get_headlines(api, ticker, days_back=days_back)
            sentiment, method = finbert_score(headlines)
            score             = composite_score(pd, sentiment, len(headlines), account_size)
            grade             = risk_grade(pd)
            results.append({
                "ticker":      ticker,
                "price":       pd["price"],
                "change_pct":  pd["change_pct"],
                "vol_ratio":   pd["vol_ratio"],
                "rs_5d":       pd["rs_5d"],
                "avg_range":   pd["avg_range"],
                "sentiment":   sentiment,
                "sent_method": method,
                "news_count":  len(headlines),
                "score":       score,
                "grade":       grade,
                "direction":   "up" if pd["change_pct"] > 0 else "down",
            })
            time.sleep(0.1)
        except Exception as e:
            print(f"  Scan error {ticker}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"=== Top 5 ({label}): {[(r['ticker'], r['grade'], r['score']) for r in top5]} ===")
    return top5

# ── Account size ──────────────────────────────────────────────

def get_account_size(api):
    try:
        return min(float(api.get_account().equity), MAX_ACCOUNT)
    except:
        return MAX_ACCOUNT

# ── Full scan entry ───────────────────────────────────────────

def run_full_scan(api):
    account_size = get_account_size(api)
    print(f"Full scan | account ${account_size:.2f}")

    # Dynamic price range
    max_price = max(min(account_size * 0.45, 10.00), 0.60)
    if   account_size > 2000: min_price = 5.00
    elif account_size > 500:  min_price = 2.00
    elif account_size > 100:  min_price = 1.00
    else:                     min_price = 0.50

    # Dynamic volume floor
    if   account_size > 5000: min_vol = 1_000_000
    elif account_size > 1000: min_vol = 500_000
    elif account_size > 200:  min_vol = 250_000
    else:                     min_vol = 100_000

    universe = build_universe(api, max_price, min_price, min_vol)

    if not universe:
        print("Universe empty — using fallback")
        universe = [{"symbol": s, "price": 1.0, "volume": 500_000}
                    for s in ["SIRI","TELL","CLOV","NKLA","MVIS",
                               "SOFI","HOOD","NIO","MARA","RIOT"]]

    today     = run_scan(api, universe, days_back=1, account_size=account_size)
    yesterday = run_scan(api, universe, days_back=2, account_size=account_size)

    return {
        "today":         today,
        "yesterday":     yesterday,
        "scanned_at":    datetime.now(NY).strftime("%I:%M %p ET"),
        "account_size":  round(account_size, 2),
        "price_range":   f"${min_price}–${max_price:.2f}",
        "universe_size": len(universe),
    }