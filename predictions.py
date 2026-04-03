"""
predictions.py — Multi-factor prediction engine

Combines:
  - 3-day news sentiment trend
  - Next-day price direction (momentum trajectory)
  - Volatility forecast (ATR expansion/contraction)
  - Pattern recognition (channel, breakout, consolidation)
  - Market condition forecast (regime + volatility proxy)
  - Earnings risk flag (avoid trading near earnings)

Output: prediction_score (-100 to +100) per ticker
  > 30  = bullish prediction  → bot gets confidence boost
  < -30 = bearish prediction  → bot skips or reduces size
  0±30  = neutral             → no adjustment
"""

import time
from datetime import datetime, timedelta
import pytz
import requests as req
import os

NY         = pytz.timezone("America/New_York")
HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ── Helpers ───────────────────────────────────────────────────

def safe_bars(api, ticker, timeframe="1Day", days=20, limit=20):
    """Fetch bars with IEX feed, handle MultiIndex, return clean DataFrame or None."""
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=days + 5)
        bars  = api.get_bars(ticker, timeframe,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=limit, feed="iex").df
        if bars is None or bars.empty:
            return None
        if hasattr(bars.index, 'levels'):
            lvl = bars.index.get_level_values(0)
            if ticker in lvl:
                bars = bars.loc[ticker]
            else:
                return None
        return bars if len(bars) >= 3 else None
    except Exception as e:
        print(f"  safe_bars error {ticker}: {e}")
        return None

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s) / len(s) if s else 0

def calc_atr(bars, period=10):
    """Average True Range over last N bars."""
    try:
        trs = []
        closes = list(bars["close"])
        highs  = list(bars["high"])
        lows   = list(bars["low"])
        for i in range(1, min(len(closes), period + 1)):
            hl  = highs[i]  - lows[i]
            hpc = abs(highs[i]  - closes[i-1])
            lpc = abs(lows[i]   - closes[i-1])
            trs.append(max(hl, hpc, lpc))
        return sum(trs) / len(trs) if trs else 0
    except:
        return 0

def keyword_score(text):
    pos = ["surge","soar","rally","beat","record","upgrade","bullish","growth",
           "profit","breakthrough","strong","exceed","outperform","expansion",
           "momentum","breakout","launch","partnership","dividend","acquire"]
    neg = ["crash","plunge","drop","miss","downgrade","bearish","loss","decline",
           "lawsuit","layoff","cut","weak","disappoint","fraud","bankruptcy",
           "halt","delisted","investigation","warning","debt","recall"]
    t = text.lower()
    return sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)

# ── Feature 1: 3-day news sentiment trend ────────────────────

def news_sentiment_trend(api, ticker):
    """
    Pulls headlines for each of the last 3 days separately.
    Returns: trend_score, direction, daily_scores
    trend_score > 0 = sentiment improving
    trend_score < 0 = sentiment worsening
    """
    daily_scores = []
    for days_ago in [3, 2, 1]:
        try:
            end   = datetime.now(pytz.utc) - timedelta(days=days_ago-1)
            start = end - timedelta(days=1)
            news  = api.get_news(ticker,
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10)
            headlines = [n.headline for n in news] if news else []

            if not headlines:
                daily_scores.append(0)
                continue

            # Use FinBERT if available, else keyword
            if HF_TOKEN and len(headlines) > 0:
                try:
                    payload  = {"inputs": headlines[:5], "options": {"wait_for_model": True}}
                    resp     = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=12)
                    if resp.status_code == 200:
                        total = 0
                        for result in resp.json():
                            sm     = {r["label"]: r["score"] for r in result}
                            total += sm.get("positive", 0) - sm.get("negative", 0)
                        daily_scores.append(round(total, 3))
                        continue
                except:
                    pass
            daily_scores.append(sum(keyword_score(h) for h in headlines))
        except Exception as e:
            print(f"  sentiment_trend error {ticker} day-{days_ago}: {e}")
            daily_scores.append(0)

    if len(daily_scores) < 2:
        return 0, "neutral", daily_scores

    # Trend = is sentiment moving up or down?
    # Compare recent day vs 3 days ago
    trend = daily_scores[-1] - daily_scores[0]
    # Normalize to -30 to +30
    normalized = max(-30, min(30, trend * 10))

    direction = "improving" if trend > 0.1 else "worsening" if trend < -0.1 else "neutral"
    return round(normalized, 2), direction, daily_scores

# ── Feature 2: Next-day price direction predictor ─────────────

def price_direction_predictor(bars):
    """
    Analyzes momentum trajectory to predict next-day direction.
    Returns: score (-40 to +40), confidence, signals_used
    """
    if bars is None or len(bars) < 5:
        return 0, "low", []

    closes  = list(bars["close"])
    volumes = list(bars["volume"])
    score   = 0
    signals = []

    # 1. Short-term momentum (last 3 days)
    if len(closes) >= 4:
        momentum_3d = (closes[-1] - closes[-4]) / closes[-4] * 100
        if   momentum_3d >  3: score += 10; signals.append(f"3d momentum +{momentum_3d:.1f}%")
        elif momentum_3d >  1: score +=  5; signals.append(f"3d momentum +{momentum_3d:.1f}%")
        elif momentum_3d < -3: score -= 10; signals.append(f"3d momentum {momentum_3d:.1f}%")
        elif momentum_3d < -1: score -=  5; signals.append(f"3d momentum {momentum_3d:.1f}%")

    # 2. Acceleration — is momentum speeding up or slowing?
    if len(closes) >= 6:
        recent_move = closes[-1] - closes[-3]
        prior_move  = closes[-3] - closes[-5]
        if recent_move > 0 and recent_move > prior_move:
            score += 8; signals.append("momentum accelerating ↑")
        elif recent_move < 0 and recent_move < prior_move:
            score -= 8; signals.append("momentum accelerating ↓")
        elif recent_move > 0 and recent_move < prior_move:
            score -= 3; signals.append("momentum decelerating")

    # 3. Volume confirmation — high volume on up days = conviction
    if len(closes) >= 3 and len(volumes) >= 3:
        avg_vol     = sum(volumes[-10:]) / len(volumes[-10:]) if len(volumes) >= 10 else sum(volumes) / len(volumes)
        last_vol    = volumes[-1]
        last_move   = closes[-1] - closes[-2]
        vol_ratio   = last_vol / avg_vol if avg_vol > 0 else 1
        if last_move > 0 and vol_ratio > 1.5:
            score += 8; signals.append(f"volume confirmation ({vol_ratio:.1f}x on up day)")
        elif last_move < 0 and vol_ratio > 1.5:
            score -= 8; signals.append(f"volume selling ({vol_ratio:.1f}x on down day)")
        elif last_move > 0 and vol_ratio < 0.7:
            score -= 3; signals.append("weak volume on up day")

    # 4. MA alignment — price above both short and long MAs = bullish
    if len(closes) >= 10:
        ma5  = calc_ma(closes, 5)
        ma10 = calc_ma(closes, 10)
        p    = closes[-1]
        if p > ma5 > ma10:
            score += 7; signals.append("price > MA5 > MA10 (bullish stack)")
        elif p < ma5 < ma10:
            score -= 7; signals.append("price < MA5 < MA10 (bearish stack)")
        elif p > ma5 and ma5 < ma10:
            score += 3; signals.append("price above MA5, recovering")

    # 5. Recent candle body — consecutive closes in same direction
    if len(closes) >= 3:
        c1 = closes[-1] > closes[-2]
        c2 = closes[-2] > closes[-3]
        if c1 and c2:
            score += 5; signals.append("2 consecutive up closes")
        elif not c1 and not c2:
            score -= 5; signals.append("2 consecutive down closes")

    score     = max(-40, min(40, score))
    confidence = "high" if abs(score) > 25 else "medium" if abs(score) > 12 else "low"
    return round(score, 1), confidence, signals

# ── Feature 3: Volatility forecast ───────────────────────────

def volatility_forecast(bars):
    """
    Detects whether volatility is expanding or contracting.
    Expanding = bigger moves expected (opportunity or risk)
    Contracting = consolidation, breakout may be coming
    Returns: score (-20 to +20), state, details
    """
    if bars is None or len(bars) < 6:
        return 0, "unknown", {}

    try:
        # Split bars into recent and prior halves
        mid   = len(bars) // 2
        prior = bars.iloc[:mid]
        recent= bars.iloc[mid:]

        atr_prior  = calc_atr(prior)
        atr_recent = calc_atr(recent)

        if atr_prior == 0:
            return 0, "unknown", {}

        expansion_ratio = atr_recent / atr_prior
        closes = list(bars["close"])
        price  = closes[-1]

        # ATR as % of price
        atr_pct = (atr_recent / price) * 100 if price > 0 else 0

        state   = "unknown"
        score   = 0
        details = {
            "atr_prior":   round(atr_prior,  3),
            "atr_recent":  round(atr_recent, 3),
            "expansion":   round(expansion_ratio, 2),
            "atr_pct":     round(atr_pct, 2),
        }

        if expansion_ratio > 1.3:
            state  = "expanding"
            # Expanding vol = bigger moves, good for momentum trades
            # But very high vol = risky for small accounts
            if atr_pct < 8:
                score = 10   # moderate expansion, good opportunity
            else:
                score = -5   # too volatile for $20 account
        elif expansion_ratio < 0.75:
            state = "contracting"
            # Tight consolidation often precedes a breakout
            score = 8  # coiled spring — breakout coming
        else:
            state = "stable"
            score = 2

        details["state"] = state
        return score, state, details

    except Exception as e:
        print(f"  volatility_forecast error: {e}")
        return 0, "unknown", {}

# ── Feature 4: Pattern recognition ───────────────────────────

def pattern_recognition(bars):
    """
    Detects basic price patterns.
    Returns: score (-25 to +25), pattern_name, description
    """
    if bars is None or len(bars) < 6:
        return 0, "insufficient_data", ""

    try:
        closes = list(bars["close"])
        highs  = list(bars["high"])
        lows   = list(bars["low"])
        n      = len(closes)

        score   = 0
        pattern = "none"
        desc    = ""

        # Higher highs + higher lows = uptrend
        if n >= 6:
            hh = highs[-1] > highs[-3] > highs[-5]
            hl = lows[-1]  > lows[-3]  > lows[-5]
            lh = highs[-1] < highs[-3] < highs[-5]
            ll = lows[-1]  < lows[-3]  < lows[-5]

            if hh and hl:
                score = 20; pattern = "uptrend"; desc = "Higher highs and higher lows"
            elif lh and ll:
                score = -20; pattern = "downtrend"; desc = "Lower highs and lower lows"

        # Consolidation — tight range over last 5 bars
        if n >= 5 and pattern == "none":
            recent_range = max(highs[-5:]) - min(lows[-5:])
            avg_price    = sum(closes[-5:]) / 5
            range_pct    = (recent_range / avg_price) * 100 if avg_price > 0 else 0
            if range_pct < 3:
                score = 5; pattern = "consolidation"; desc = f"Tight range ({range_pct:.1f}%) — breakout likely"

        # Breakout — price closes above recent resistance
        if n >= 8 and pattern in ("none", "consolidation"):
            resistance   = max(highs[-8:-2])
            support      = min(lows[-8:-2])
            current      = closes[-1]
            if current > resistance * 1.01:
                score = 25; pattern = "breakout_up"; desc = f"Broke above resistance ${resistance:.2f}"
            elif current < support * 0.99:
                score = -25; pattern = "breakdown"; desc = f"Broke below support ${support:.2f}"

        # V-shape recovery — sharp drop followed by recovery
        if n >= 6 and pattern == "none":
            mid_low   = min(closes[-6:-2])
            start_p   = closes[-6]
            current   = closes[-1]
            drop_pct  = (start_p - mid_low) / start_p * 100 if start_p > 0 else 0
            recov_pct = (current - mid_low) / mid_low * 100 if mid_low > 0 else 0
            if drop_pct > 5 and recov_pct > 4:
                score = 15; pattern = "v_recovery"; desc = f"V-shape recovery after {drop_pct:.1f}% drop"

        return max(-25, min(25, score)), pattern, desc

    except Exception as e:
        print(f"  pattern_recognition error: {e}")
        return 0, "error", ""

# ── Feature 5: Earnings risk flag ────────────────────────────

def earnings_risk(api, ticker):
    """
    Flags if a stock has had unusual news volume in the last 2 days
    (proxy for earnings/catalyst since free API has no earnings calendar).
    Returns: risk_level, score_adjustment
    """
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=2)
        news  = api.get_news(ticker,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=20)
        count = len(news) if news else 0

        # Check for earnings keywords in headlines
        earnings_words = ["earnings","eps","revenue","quarterly","guidance",
                          "results","beat","miss","q1","q2","q3","q4"]
        earnings_hit = 0
        if news:
            for n in news:
                if any(w in n.headline.lower() for w in earnings_words):
                    earnings_hit += 1

        if earnings_hit >= 2:
            return "high", -20  # likely near earnings — avoid
        elif earnings_hit == 1 or count > 15:
            return "medium", -8  # elevated activity — be cautious
        else:
            return "low", 0

    except Exception as e:
        print(f"  earnings_risk error {ticker}: {e}")
        return "unknown", 0

# ── Feature 6: Market condition forecast ─────────────────────

def market_condition_forecast(api, regime):
    """
    Uses regime + a volatility proxy to forecast overall market conditions.
    Returns: condition, score (-15 to +15)
    """
    try:
        # Use UVXY (2x VIX ETF) as volatility proxy — trades on IEX
        bars = safe_bars(api, "UVXY", days=10, limit=10)
        if bars is not None and len(bars) >= 5:
            closes   = list(bars["close"])
            vix_move = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] > 0 else 0
            # Rising UVXY = rising fear = bad for market
            vix_rising = vix_move > 5
            vix_falling = vix_move < -5
        else:
            vix_rising  = False
            vix_falling = False
            vix_move    = 0

        score     = 0
        condition = "neutral"

        if regime == "trending_up" and vix_falling:
            score = 15; condition = "bullish"
        elif regime == "trending_up" and not vix_rising:
            score = 8;  condition = "mildly_bullish"
        elif regime == "trending_down" and vix_rising:
            score = -15; condition = "bearish"
        elif regime == "trending_down" and not vix_falling:
            score = -8;  condition = "mildly_bearish"
        elif regime == "ranging":
            score = 2;  condition = "neutral"
        elif vix_rising:
            score = -10; condition = "elevated_fear"
        elif vix_falling:
            score = 5;  condition = "calming"

        return condition, score, round(vix_move, 1)

    except Exception as e:
        print(f"  market_condition_forecast error: {e}")
        return "unknown", 0, 0

# ── Master prediction function ────────────────────────────────

def predict_ticker(api, ticker, regime="ranging"):
    """
    Runs all prediction features for a single ticker.
    Returns a full prediction dict with composite score.
    Score: -100 to +100
      > 30  = bullish — bot gets size boost
      < -30 = bearish — bot skips or cuts size
    """
    result = {
        "ticker":         ticker,
        "score":          0,
        "label":          "neutral",
        "confidence":     "low",
        "components":     {},
        "signals":        [],
        "timestamp":      datetime.now(NY).strftime("%I:%M %p ET"),
    }

    try:
        # Fetch bars once, reuse across features
        bars = safe_bars(api, ticker, days=20, limit=20)

        # Run all features
        sent_score,  sent_dir,    sent_daily  = news_sentiment_trend(api, ticker)
        dir_score,   dir_conf,    dir_signals = price_direction_predictor(bars)
        vol_score,   vol_state,   vol_details = volatility_forecast(bars)
        pat_score,   pat_name,    pat_desc    = pattern_recognition(bars)
        earn_risk,   earn_adj                 = earnings_risk(api, ticker)
        mkt_cond,    mkt_score,   vix_move    = market_condition_forecast(api, regime)

        # Composite score
        total = (
            sent_score  * 1.0 +   # sentiment trend weight
            dir_score   * 1.0 +   # price direction weight
            vol_score   * 0.5 +   # volatility weight (less direct)
            pat_score   * 0.8 +   # pattern weight
            earn_adj    * 1.0 +   # earnings risk (hard penalty)
            mkt_score   * 0.7     # market condition weight
        )
        total = max(-100, min(100, round(total, 1)))

        # Label
        if   total > 50:  label = "strong_buy"
        elif total > 25:  label = "bullish"
        elif total > 0:   label = "mildly_bullish"
        elif total > -25: label = "mildly_bearish"
        elif total > -50: label = "bearish"
        else:             label = "strong_avoid"

        # Confidence from direction predictor + pattern strength
        if dir_conf == "high" and abs(pat_score) > 15:
            confidence = "high"
        elif dir_conf == "medium" or abs(pat_score) > 8:
            confidence = "medium"
        else:
            confidence = "low"

        result.update({
            "score":      total,
            "label":      label,
            "confidence": confidence,
            "components": {
                "sentiment_trend": {
                    "score":       sent_score,
                    "direction":   sent_dir,
                    "daily":       sent_daily,
                },
                "price_direction": {
                    "score":       dir_score,
                    "confidence":  dir_conf,
                    "signals":     dir_signals,
                },
                "volatility": {
                    "score":       vol_score,
                    "state":       vol_state,
                    "details":     vol_details,
                },
                "pattern": {
                    "score":       pat_score,
                    "name":        pat_name,
                    "description": pat_desc,
                },
                "earnings_risk": {
                    "level":       earn_risk,
                    "adjustment":  earn_adj,
                },
                "market_condition": {
                    "condition":   mkt_cond,
                    "score":       mkt_score,
                    "vix_move":    vix_move,
                },
            },
            "signals": dir_signals,
        })

    except Exception as e:
        print(f"predict_ticker error {ticker}: {e}")
        result["score"] = 0
        result["label"] = "neutral"

    print(f"  Prediction {ticker}: {result['score']:+.0f} ({result['label']}) conf={result['confidence']}")
    return result

def run_predictions(api, tickers, regime="ranging"):
    """Run predictions for a list of tickers. Called by bot.py and scanner."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = predict_ticker(api, ticker, regime)
            time.sleep(0.5)  # gentle rate limit
        except Exception as e:
            print(f"  run_predictions error {ticker}: {e}")
            results[ticker] = {"ticker": ticker, "score": 0, "label": "neutral", "confidence": "low"}
    return results