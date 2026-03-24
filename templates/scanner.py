import time
from datetime import datetime, timedelta
import pytz

# ── FinBERT sentiment via HuggingFace inference API ──────────
# Free tier: ~30k requests/month. No GPU needed.
# Set HUGGINGFACE_TOKEN in Render environment variables.
import os, requests as req

HF_TOKEN  = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL    = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

UNIVERSE = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","NFLX","ORCL",
    "JPM","BAC","GS","V","MA","PYPL","COIN","SQ","HOOD",
    "XOM","CVX","OXY","LNG",
    "PLTR","SOFI","RIVN","LCID","NIO","MARA","RIOT",
    "SPY","QQQ","IWM"
]

# ── Fallback keyword scorer (used if HuggingFace is unavailable) ──

POSITIVE_WORDS = [
    "surge","soar","rally","beat","record","upgrade","buy","bullish","growth",
    "profit","revenue","partnership","launch","breakthrough","strong","exceed",
    "outperform","raise","acquire","expansion","dividend"
]
NEGATIVE_WORDS = [
    "crash","plunge","drop","miss","downgrade","sell","bearish","loss","decline",
    "lawsuit","recall","layoff","cut","weak","disappoint","probe","fraud",
    "bankruptcy","debt","warning","risk"
]

def keyword_score(text):
    text = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in text: score += 1
    for w in NEGATIVE_WORDS:
        if w in text: score -= 1
    return score

# ── FinBERT sentiment scorer ──────────────────────────────────

def finbert_score(headlines):
    """
    Sends headlines to FinBERT via HuggingFace Inference API.
    Returns a float: positive = bullish, negative = bearish.
    Falls back to keyword scoring if API is unavailable.
    """
    if not headlines:
        return 0, "no_news"

    if not HF_TOKEN:
        # Fallback to keyword scoring
        scores = [keyword_score(h) for h in headlines]
        return sum(scores), "keyword"

    try:
        # Batch headlines into one request (max 5 at a time for free tier)
        batch    = headlines[:5]
        payload  = {"inputs": batch, "options": {"wait_for_model": True}}
        response = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=15)

        if response.status_code != 200:
            print(f"  FinBERT API error {response.status_code} — falling back to keywords")
            scores = [keyword_score(h) for h in headlines]
            return sum(scores), "keyword_fallback"

        results = response.json()
        total   = 0
        for result in results:
            # Each result is a list of {label, score}
            scores_map = {r["label"]: r["score"] for r in result}
            pos = scores_map.get("positive", 0)
            neg = scores_map.get("negative", 0)
            total += (pos - neg)  # net sentiment per headline

        return round(total, 3), "finbert"

    except Exception as e:
        print(f"  FinBERT error: {e} — falling back to keywords")
        scores = [keyword_score(h) for h in headlines]
        return sum(scores), "keyword_fallback"

# ── News fetcher ──────────────────────────────────────────────

def get_headlines(api, ticker, days_back=1):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days_back)
        news  = api.get_news(
            ticker,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=10
        )
        return [n.headline for n in news] if news else []
    except Exception as e:
        print(f"  News fetch error {ticker}: {e}")
        return []

# ── Price momentum ────────────────────────────────────────────

def get_price_data(api, ticker):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=10)
        bars  = api.get_bars(
            ticker, "1Day",
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=10
        ).df
        if bars.empty or len(bars) < 2:
            return None

        latest     = bars.iloc[-1]
        prev       = bars.iloc[-2]
        price      = round(float(latest["close"]), 2)
        prev_close = round(float(prev["close"]), 2)
        change_pct = round((price - prev_close) / prev_close * 100, 2)
        avg_vol    = bars["volume"].mean()
        vol_ratio  = round(float(latest["volume"]) / avg_vol, 2) if avg_vol > 0 else 1.0

        # Relative strength: 5-day return
        five_day_ago = float(bars.iloc[max(0, len(bars)-5)]["close"])
        rs_5d = round((price - five_day_ago) / five_day_ago * 100, 2)

        return {
            "price":      price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "vol_ratio":  vol_ratio,
            "rs_5d":      rs_5d,
        }
    except Exception as e:
        print(f"  Bar error {ticker}: {e}")
        return None

# ── Composite scorer ──────────────────────────────────────────

def composite_score(price_data, sentiment, news_count):
    """
    Score components:
    - Momentum:  absolute change_pct (volatility = opportunity)
    - Volume:    surge above average (institutional conviction)
    - Sentiment: FinBERT net score (quality-adjusted news signal)
    - RS 5d:     relative strength over 5 days
    - Coverage:  number of articles (attention = movement)
    """
    s = 0
    s += abs(price_data["change_pct"]) * 2.5
    s += abs(price_data["rs_5d"])      * 1.5

    vr = price_data["vol_ratio"]
    if   vr > 3.0: s += 20
    elif vr > 2.0: s += 12
    elif vr > 1.5: s += 6
    elif vr > 1.2: s += 3

    s += sentiment * 8           # FinBERT signal weighted heavily
    s += min(news_count * 2, 12) # coverage bonus capped at 12

    return round(s, 2)

# ── Main scan ─────────────────────────────────────────────────

def run_scan(api, days_back=1):
    label = "today" if days_back == 1 else "yesterday"
    print(f"\n=== Scanner ({label}) — {len(UNIVERSE)} stocks ===")
    results = []

    for ticker in UNIVERSE:
        try:
            pd = get_price_data(api, ticker)
            if not pd:
                continue

            headlines = get_headlines(api, ticker, days_back=days_back)
            sentiment, method = finbert_score(headlines)

            score = composite_score(pd, sentiment, len(headlines))
            results.append({
                "ticker":     ticker,
                "price":      pd["price"],
                "change_pct": pd["change_pct"],
                "vol_ratio":  pd["vol_ratio"],
                "rs_5d":      pd["rs_5d"],
                "sentiment":  sentiment,
                "sent_method": method,
                "news_count": len(headlines),
                "score":      score,
                "direction":  "up" if pd["change_pct"] > 0 else "down",
            })
            print(f"  {ticker}: ${pd['price']} | {pd['change_pct']:+.1f}% | "
                  f"vol {pd['vol_ratio']}x | sent {sentiment:+.2f} ({method}) | score {score}")
            time.sleep(0.3)

        except Exception as e:
            print(f"  Scan error {ticker}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"=== Top 5 ({label}): {[r['ticker'] for r in top5]} ===\n")
    return top5

def run_full_scan(api):
    today     = run_scan(api, days_back=1)
    yesterday = run_scan(api, days_back=2)
    return {
        "today":      today,
        "yesterday":  yesterday,
        "scanned_at": datetime.now(pytz.timezone("America/New_York")).strftime("%I:%M %p ET")
    }
