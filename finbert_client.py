"""
finbert_client.py — Rate-limited FinBERT sentiment scoring.

HuggingFace Inference API free tier: FINBERT_DAILY_LIMIT calls per day.
After the limit is hit, all calls fall back to keyword scoring automatically.

Set HUGGINGFACE_TOKEN environment variable to enable FinBERT.
Both scanner.py and predictions.py import from here to share the daily counter.
"""

import os
import requests as req
from datetime import date

HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

FINBERT_DAILY_LIMIT = 3   # free tier — 3 calls per day

POSITIVE_WORDS = [
    "surge","soar","rally","beat","record","upgrade","buy","bullish","growth",
    "profit","revenue","partnership","launch","breakthrough","strong","exceed",
    "outperform","raise","acquire","expansion","dividend","momentum","breakout",
]
NEGATIVE_WORDS = [
    "crash","plunge","drop","miss","downgrade","sell","bearish","loss","decline",
    "lawsuit","recall","layoff","cut","weak","disappoint","probe","fraud",
    "bankruptcy","debt","warning","risk","halt","delisted","investigation",
]

_counter = {"date": None, "n": 0}


def _reset_if_new_day():
    today = date.today()
    if _counter["date"] != today:
        _counter.update({"date": today, "n": 0})


def calls_remaining() -> int:
    _reset_if_new_day()
    return max(0, FINBERT_DAILY_LIMIT - _counter["n"])


def keyword_score(text: str) -> float:
    t = text.lower()
    return (sum(1 for w in POSITIVE_WORDS if w in t) -
            sum(1 for w in NEGATIVE_WORDS if w in t))


def finbert_score(headlines: list) -> tuple:
    """
    Score a list of headlines with FinBERT.

    Returns (score, method) where method is one of:
      "finbert"          — API call succeeded, score is FinBERT output
      "keyword"          — no token, limit reached, or empty headlines
      "keyword_fallback" — token set but API returned an error

    Each successful API call counts against the daily limit.
    Once FINBERT_DAILY_LIMIT is reached, all remaining calls use keywords.
    """
    if not headlines:
        return 0, "no_news"

    kw = sum(keyword_score(h) for h in headlines)

    if not HF_TOKEN:
        return kw, "keyword"

    _reset_if_new_day()
    if _counter["n"] >= FINBERT_DAILY_LIMIT:
        return kw, "keyword"

    try:
        payload = {"inputs": headlines[:5], "options": {"wait_for_model": True}}
        resp    = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"  FinBERT HTTP {resp.status_code} — keyword fallback")
            return kw, "keyword_fallback"

        _counter["n"] += 1
        remaining = FINBERT_DAILY_LIMIT - _counter["n"]
        print(f"  FinBERT: scored {len(headlines)} headlines "
              f"({remaining} call{'s' if remaining != 1 else ''} remaining today)")

        total = 0
        for result in resp.json():
            sm     = {r["label"]: r["score"] for r in result}
            total += sm.get("positive", 0) - sm.get("negative", 0)
        return round(total, 3), "finbert"

    except Exception as e:
        print(f"  FinBERT error: {e}")
        return kw, "keyword_fallback"
