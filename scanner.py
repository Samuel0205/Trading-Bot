import time
from datetime import datetime, timedelta
import pytz
import os
import requests as req

HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

UNIVERSE = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","NFLX","ORCL",
    "JPM","BAC","GS","V","MA","PYPL","COIN","SQ","HOOD",
    "XOM","CVX","OXY","LNG",
    "PLTR","SOFI","RIVN","LCID","NIO","MARA","RIOT",
    "SPY","QQQ","IWM"
]

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

def finbert_score(headlines):
    if not headlines:
        return 0, "no_news"
    if not HF_TOKEN:
        scores = [keyword_score(h) for h in headlines]
        return sum(scores), "keyword"
    try:
        batch    = headlines[:5]
        payload  = {"inputs": batch, "options": {"wait_for_model": True}}
        response = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"  FinBERT error {response.status_code} — using keywords")
            return sum(keyword_score(h) for h in headlines), "keyword_fallback"
        results = response.json()
        total   = 0
        for result in results:
            scores_map = {r["label"]: r["score"] for r in result}
            total += scores_map.get("positive", 0) - scores_map.get("negative", 0)
        return round(total, 3), "finbert"
    except Exception as e:
        print(f"  FinBERT error: {e}")
        return sum(keyword_score(h) for h in headlines), "keyword_fallback"

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
        print(f"  News error {ticker}: {e}")
        return []

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
        five_day   = float(bars.iloc[max(0, len(bars)-5)]["close"])
        rs_5d      = round((price - five_day) / five_day * 100, 2)
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

def composite_score(price_data, sentiment, news_count):
    s = 0
    s += abs(price_data["change_pct"]) * 2.5
    s += abs(price_data["rs_5d"])      * 1.5
    vr = price_data["vol_ratio"]
    if   vr > 3.0: s += 20
    elif vr > 2.0: s += 12
    elif vr > 1.5: s += 6
    elif vr > 1.2: s += 3
    s += sentiment * 8
    s += min(news_count * 2, 12)
    return round(s, 2)

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
                "ticker":      ticker,
                "price":       pd["price"],
                "change_pct":  pd["change_pct"],
                "vol_ratio":   pd["vol_ratio"],
                "rs_5d":       pd["rs_5d"],
                "sentiment":   sentiment,
                "sent_method": method,
                "news_count":  len(headlines),
                "score":       score,
                "direction":   "up" if pd["change_pct"] > 0 else "down",
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