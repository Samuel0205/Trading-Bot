import time, os, requests as req
from datetime import datetime, timedelta
import pytz

HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

NY = pytz.timezone("America/New_York")

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
        response = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=15)
        if response.status_code != 200:
            return sum(keyword_score(h) for h in headlines), "keyword_fallback"
        total = 0
        for result in response.json():
            sm = {r["label"]: r["score"] for r in result}
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

# ── Dynamic universe builder ──────────────────────────────────

def build_universe(api, max_price, min_price=0.50, min_volume=100_000):
    """
    Dynamically builds a tradeable universe based on what we can
    actually afford. Pulls active NYSE/NASDAQ assets, filters by
    price range and volume. Self-scales as account grows.
    """
    print(f"  Building universe: price ${min_price}–${max_price}, vol >{min_volume:,}")
    try:
        # Pull all active US equity assets
        assets = api.list_assets(status="active", asset_class="us_equity")
        tradeable = [
            a for a in assets
            if a.tradable
            and a.exchange in ("NYSE", "NASDAQ", "ARCA")
            and not a.symbol.endswith(("W", "R", "P", "Q"))  # skip warrants/rights/OTC
            and len(a.symbol) <= 5
        ]
        print(f"  {len(tradeable)} tradeable assets found")

        # Batch price check — get latest bars for candidates
        symbols  = [a.symbol for a in tradeable]
        universe = []
        batch_size = 100  # Alpaca allows up to 100 per request

        for i in range(0, min(len(symbols), 1000), batch_size):
            batch = symbols[i:i+batch_size]
            try:
                end   = datetime.now(pytz.utc)
                start = end - timedelta(days=5)
                bars  = api.get_bars(
                    batch, "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=5
                ).df

                if bars.empty:
                    continue

                # bars.df has MultiIndex (symbol, timestamp)
                for sym in batch:
                    try:
                        if sym not in bars.index.get_level_values(0):
                            continue
                        sym_bars = bars.loc[sym]
                        if len(sym_bars) < 2:
                            continue
                        latest  = sym_bars.iloc[-1]
                        price   = float(latest["close"])
                        avg_vol = sym_bars["volume"].mean()

                        if (min_price <= price <= max_price
                                and avg_vol >= min_volume):
                            universe.append({
                                "symbol": sym,
                                "price":  round(price, 2),
                                "volume": int(avg_vol)
                            })
                    except:
                        continue
                time.sleep(0.2)
            except Exception as e:
                print(f"  Batch error: {e}")
                continue

        print(f"  Universe built: {len(universe)} stocks in ${min_price}–${max_price} range")
        return universe

    except Exception as e:
        print(f"  build_universe error: {e}")
        return []

# ── Price data ────────────────────────────────────────────────

def get_price_data(api, ticker):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=10)
        bars  = api.get_bars(ticker, "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=10).df
        if bars.empty or len(bars) < 2:
            return None

        # Handle MultiIndex if returned
        if hasattr(bars.index, 'levels'):
            bars = bars.loc[ticker] if ticker in bars.index.get_level_values(0) else bars

        latest     = bars.iloc[-1]
        prev       = bars.iloc[-2]
        price      = round(float(latest["close"]), 2)
        prev_close = round(float(prev["close"]), 2)
        change_pct = round((price - prev_close) / prev_close * 100, 2)
        avg_vol    = bars["volume"].mean()
        vol_ratio  = round(float(latest["volume"]) / avg_vol, 2) if avg_vol > 0 else 1.0
        five_day   = float(bars.iloc[max(0, len(bars)-5)]["close"])
        rs_5d      = round((price - five_day) / five_day * 100, 2)

        # Intraday volatility — range as % of price
        avg_range  = ((bars["high"] - bars["low"]) / bars["close"]).mean() * 100

        return {
            "price":      price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "vol_ratio":  vol_ratio,
            "rs_5d":      rs_5d,
            "avg_range":  round(float(avg_range), 2),
        }
    except Exception as e:
        print(f"  Bar error {ticker}: {e}")
        return None

# ── Composite scorer ──────────────────────────────────────────

def composite_score(pd, sentiment, news_count, account_size):
    """
    Scores a stock for tradability with small account.
    Weights shift as account grows — more conservative over time.
    """
    s = 0

    # Momentum
    s += abs(pd["change_pct"]) * 2.5
    s += abs(pd["rs_5d"])      * 1.5

    # Volume conviction
    vr = pd["vol_ratio"]
    if   vr > 5.0: s += 25
    elif vr > 3.0: s += 18
    elif vr > 2.0: s += 10
    elif vr > 1.5: s +=  5
    elif vr > 1.2: s +=  2

    # Sentiment
    s += sentiment * 8
    s += min(news_count * 2, 12)

    # Intraday range bonus — more volatile = more opportunity for small accounts
    if account_size < 100:
        s += pd["avg_range"] * 2   # favor volatile stocks when small
    elif account_size < 500:
        s += pd["avg_range"] * 1   # balanced
    # over $500 — don't bonus volatility, focus on quality signals

    # Price sweet spot bonus — cheaper = more shares = better for tiny accounts
    if account_size < 50:
        if pd["price"] < 2:   s += 10
        elif pd["price"] < 5: s += 5
    elif account_size < 200:
        if pd["price"] < 5:   s += 8
        elif pd["price"] < 8: s += 3

    return round(s, 2)

# ── Risk grade ────────────────────────────────────────────────

def risk_grade(pd):
    """
    Returns a letter grade A–F based on how safe the stock looks.
    Shown on dashboard to help manual review.
    """
    score = 0
    if pd["vol_ratio"] < 3:      score += 2
    if abs(pd["change_pct"]) < 5: score += 2
    if abs(pd["rs_5d"]) < 10:    score += 2
    if pd["avg_range"] < 5:      score += 2
    if pd["avg_range"] < 3:      score += 1
    grades = {9: "A", 8: "A", 7: "B", 6: "B", 5: "C", 4: "C", 3: "D"}
    return grades.get(score, "F")

# ── Main scan ─────────────────────────────────────────────────

def run_scan(api, universe, days_back=1, account_size=20):
    label = "today" if days_back == 1 else "yesterday"
    print(f"\n=== Scanning {len(universe)} stocks ({label}) | account ${account_size:.2f} ===")
    results = []

    for stock in universe:
        ticker = stock["symbol"]
        try:
            pd = get_price_data(api, ticker)
            if not pd:
                continue

            # Double-check price is still affordable
            if pd["price"] > (account_size * 0.6):
                continue

            headlines        = get_headlines(api, ticker, days_back=days_back)
            sentiment, method = finbert_score(headlines)
            score            = composite_score(pd, sentiment, len(headlines), account_size)
            grade            = risk_grade(pd)

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
            time.sleep(0.2)

        except Exception as e:
            print(f"  Scan error {ticker}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"=== Top 5 ({label}): {[(r['ticker'], r['grade'], r['score']) for r in top5]} ===\n")
    return top5

MAX_ACCOUNT = 20.00  # your real capital limit — update as you grow

def get_account_size(api):
    """
    Returns the SMALLER of MAX_ACCOUNT or actual equity.
    Keeps scanner scaled to your real capital, not paper balance.
    """
    try:
        return min(float(api.get_account().equity), MAX_ACCOUNT)
    except:
        return MAX_ACCOUNT

def run_full_scan(api):
    account_size = get_account_size(api)
    print(f"Account size: ${account_size:.2f}")

    # Self-scaling price ceiling — never look at stocks we can't afford
    # Always keep at least 2 shares worth of room
    max_price = min(account_size * 0.45, 10.00)  # hard cap $10 for now
    max_price = max(max_price, 0.60)              # always at least $0.60 ceiling
    min_price = 0.50

    # As account grows past $100, raise the floor to avoid penny stocks
    if account_size > 100:  min_price = 1.00
    if account_size > 500:  min_price = 2.00
    if account_size > 2000: min_price = 5.00

    # Dynamic volume floor — require more liquidity as account grows
    min_vol = 100_000
    if account_size > 200:  min_vol = 250_000
    if account_size > 1000: min_vol = 500_000
    if account_size > 5000: min_vol = 1_000_000

    # Build fresh universe based on what we can afford right now
    universe = build_universe(api, max_price, min_price, min_vol)

    if not universe:
        print("Universe empty — using hardcoded fallback")
        universe = [
            {"symbol": s, "price": 1.0, "volume": 500_000}
            for s in ["SIRI","TELL","CLOV","NKLA","MVIS"]
        ]

    today     = run_scan(api, universe, days_back=1, account_size=account_size)
    yesterday = run_scan(api, universe, days_back=2, account_size=account_size)

    return {
        "today":        today,
        "yesterday":    yesterday,
        "scanned_at":   datetime.now(NY).strftime("%I:%M %p ET"),
        "account_size": round(account_size, 2),
        "price_range":  f"${min_price}–${max_price:.2f}",
        "universe_size": len(universe),
    }