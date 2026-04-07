"""
predictions.py — Multi-factor prediction engine

Features:
  - 3-day news sentiment trend (FinBERT or keyword fallback)
  - Next-day price direction (momentum trajectory + acceleration)
  - Volatility forecast (ATR expansion/contraction)
  - Pattern recognition (channels, breakouts, V-recovery)
  - Earnings risk flag
  - Market condition forecast (regime + fear proxy)
  - Multi-timeframe confirmation (daily trend vs intraday)

Output: prediction_score (-100 to +100) per ticker
  > 30  = bullish  → bot gets confidence boost + larger size
  < -30 = bearish  → bot skips trade
  0±30  = neutral  → normal behavior
"""

import time, os
import requests as req
from datetime import datetime, timedelta
import pytz

NY         = pytz.timezone("America/New_York")
HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")
HF_URL     = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ── Helpers ───────────────────────────────────────────────────

def safe_bars(api, ticker, timeframe="1Day", days=20, limit=20):
    """Fetch bars with IEX feed. Returns clean DataFrame or None."""
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=days + 5)
        bars  = api.get_bars(
            ticker, timeframe,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=limit, feed="iex"
        ).df
        if bars is None or bars.empty:
            return None
        if hasattr(bars.index, 'levels'):
            if ticker in bars.index.get_level_values(0):
                bars = bars.loc[ticker]
            else:
                return None
        return bars if len(bars) >= 3 else None
    except Exception as e:
        print(f"  safe_bars error {ticker} {timeframe}: {e}")
        return None

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s) / len(s) if s else 0

def calc_atr(bars, period=10):
    try:
        closes = list(bars["close"])
        highs  = list(bars["high"])
        lows   = list(bars["low"])
        trs    = []
        for i in range(1, min(len(closes), period + 1)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i]  - closes[i-1]),
                abs(lows[i]   - closes[i-1])
            ))
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

# ── Feature 1: Multi-timeframe confirmation ───────────────────

def multi_timeframe_analysis(api, ticker):
    """
    Checks daily trend direction.
    Returns: trend_bias (+1 bullish, -1 bearish, 0 neutral), details
    This is used to filter out trades that go against the daily trend.
    """
    bars = safe_bars(api, ticker, timeframe="1Day", days=15, limit=15)
    if bars is None or len(bars) < 5:
        return 0, "insufficient_data"

    closes = list(bars["close"])
    ma5    = calc_ma(closes, min(5,  len(closes)))
    ma10   = calc_ma(closes, min(10, len(closes)))
    latest = closes[-1]

    # Daily higher highs / lower lows
    highs = list(bars["high"])
    lows  = list(bars["low"])
    hh = len(highs) >= 4 and highs[-1] > highs[-3]
    ll = len(lows)  >= 4 and lows[-1]  < lows[-3]
    hl = len(lows)  >= 4 and lows[-1]  > lows[-3]

    if ma5 > ma10 * 1.01 and latest > ma5 and hh:
        return 1, f"daily_uptrend (ma5={ma5:.2f} > ma10={ma10:.2f})"
    elif ma5 < ma10 * 0.99 and latest < ma5 and ll:
        return -1, f"daily_downtrend (ma5={ma5:.2f} < ma10={ma10:.2f})"
    elif ma5 > ma10 and hl:
        return 1, "mild_uptrend"
    elif ma5 < ma10 and not hl:
        return -1, "mild_downtrend"
    else:
        return 0, f"ranging (ma5={ma5:.2f} ≈ ma10={ma10:.2f})"

# ── Feature 2: 3-day news sentiment trend ────────────────────

def news_sentiment_trend(api, ticker):
    """
    Returns trend direction of news sentiment over 3 days.
    Improving = bullish signal. Worsening = bearish signal.
    """
    daily_scores = []
    for days_ago in [3, 2, 1]:
        try:
            end   = datetime.now(pytz.utc) - timedelta(days=days_ago - 1)
            start = end - timedelta(days=1)
            news  = api.get_news(ticker,
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10)
            headlines = [n.headline for n in news] if news else []
            if not headlines:
                daily_scores.append(0)
                continue
            if HF_TOKEN:
                try:
                    payload = {"inputs": headlines[:5], "options": {"wait_for_model": True}}
                    resp    = req.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=12)
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
        except:
            daily_scores.append(0)

    if len(daily_scores) < 2:
        return 0, "neutral", daily_scores

    trend      = daily_scores[-1] - daily_scores[0]
    normalized = max(-30, min(30, trend * 10))
    direction  = "improving" if trend > 0.1 else "worsening" if trend < -0.1 else "neutral"
    return round(normalized, 2), direction, daily_scores

# ── Feature 3: Price direction predictor ─────────────────────

def price_direction_predictor(bars):
    """
    Momentum trajectory — is price accelerating or decelerating?
    Returns score (-40 to +40), confidence, signals used.
    """
    if bars is None or len(bars) < 5:
        return 0, "low", []

    closes  = list(bars["close"])
    volumes = list(bars["volume"])
    score   = 0
    signals = []

    # 3-day momentum
    if len(closes) >= 4:
        mom = (closes[-1] - closes[-4]) / closes[-4] * 100 if closes[-4] > 0 else 0
        if   mom >  3: score += 10; signals.append(f"3d momentum +{mom:.1f}%")
        elif mom >  1: score +=  5; signals.append(f"3d momentum +{mom:.1f}%")
        elif mom < -3: score -= 10; signals.append(f"3d momentum {mom:.1f}%")
        elif mom < -1: score -=  5; signals.append(f"3d momentum {mom:.1f}%")

    # Momentum acceleration
    if len(closes) >= 6:
        recent = closes[-1] - closes[-3]
        prior  = closes[-3] - closes[-5]
        if recent > 0 and recent > prior:
            score += 8; signals.append("momentum accelerating ↑")
        elif recent < 0 and recent < prior:
            score -= 8; signals.append("momentum accelerating ↓")
        elif recent > 0 and recent < prior:
            score -= 3; signals.append("momentum decelerating")

    # Volume confirmation
    if len(volumes) >= 3:
        avg_vol  = sum(volumes[-10:]) / min(len(volumes), 10)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        last_move = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        if last_move > 0 and vol_ratio > 1.5:
            score += 8; signals.append(f"volume confirmation {vol_ratio:.1f}x")
        elif last_move < 0 and vol_ratio > 1.5:
            score -= 8; signals.append(f"volume selling {vol_ratio:.1f}x")
        elif last_move > 0 and vol_ratio < 0.7:
            score -= 3; signals.append("weak volume on up day")

    # MA alignment
    if len(closes) >= 10:
        ma5  = calc_ma(closes, 5)
        ma10 = calc_ma(closes, 10)
        p    = closes[-1]
        if p > ma5 > ma10:
            score += 7; signals.append("price > MA5 > MA10")
        elif p < ma5 < ma10:
            score -= 7; signals.append("price < MA5 < MA10")

    # Consecutive closes
    if len(closes) >= 3:
        if closes[-1] > closes[-2] > closes[-3]:
            score += 5; signals.append("2 consecutive up closes")
        elif closes[-1] < closes[-2] < closes[-3]:
            score -= 5; signals.append("2 consecutive down closes")

    score      = max(-40, min(40, score))
    confidence = "high" if abs(score) > 25 else "medium" if abs(score) > 12 else "low"
    return round(score, 1), confidence, signals

# ── Feature 4: Volatility forecast ───────────────────────────

def volatility_forecast(bars):
    """ATR expansion = bigger moves. Contraction = breakout building."""
    if bars is None or len(bars) < 6:
        return 0, "unknown", {}
    try:
        mid         = len(bars) // 2
        atr_prior   = calc_atr(bars.iloc[:mid])
        atr_recent  = calc_atr(bars.iloc[mid:])
        if atr_prior == 0:
            return 0, "unknown", {}
        ratio   = atr_recent / atr_prior
        price   = float(bars.iloc[-1]["close"])
        atr_pct = (atr_recent / price) * 100 if price > 0 else 0
        details = {
            "atr_prior":  round(atr_prior,  3),
            "atr_recent": round(atr_recent, 3),
            "expansion":  round(ratio, 2),
            "atr_pct":    round(atr_pct, 2),
        }
        if ratio > 1.3:
            state = "expanding"
            score = 10 if atr_pct < 8 else -5
        elif ratio < 0.75:
            state = "contracting"; score = 8
        else:
            state = "stable";      score = 2
        details["state"] = state
        return score, state, details
    except Exception as e:
        print(f"  volatility_forecast error: {e}")
        return 0, "unknown", {}

# ── Feature 5: Pattern recognition ───────────────────────────

def pattern_recognition(bars):
    if bars is None or len(bars) < 6:
        return 0, "insufficient_data", ""
    try:
        closes = list(bars["close"])
        highs  = list(bars["high"])
        lows   = list(bars["low"])
        n      = len(closes)
        score  = 0; pattern = "none"; desc = ""

        if n >= 6:
            hh = highs[-1] > highs[-3] > highs[-5]
            hl = lows[-1]  > lows[-3]  > lows[-5]
            lh = highs[-1] < highs[-3] < highs[-5]
            ll = lows[-1]  < lows[-3]  < lows[-5]
            if hh and hl:
                score = 20; pattern = "uptrend";   desc = "Higher highs and higher lows"
            elif lh and ll:
                score = -20; pattern = "downtrend"; desc = "Lower highs and lower lows"

        if n >= 5 and pattern == "none":
            rng     = max(highs[-5:]) - min(lows[-5:])
            avg_p   = sum(closes[-5:]) / 5
            rng_pct = (rng / avg_p) * 100 if avg_p > 0 else 0
            if rng_pct < 3:
                score = 5; pattern = "consolidation"
                desc  = f"Tight range {rng_pct:.1f}% — breakout likely"

        if n >= 8 and pattern in ("none", "consolidation"):
            resist  = max(highs[-8:-2])
            support = min(lows[-8:-2])
            current = closes[-1]
            if current > resist * 1.01:
                score = 25; pattern = "breakout_up"
                desc  = f"Broke above resistance ${resist:.2f}"
            elif current < support * 0.99:
                score = -25; pattern = "breakdown"
                desc  = f"Broke below support ${support:.2f}"

        if n >= 6 and pattern == "none":
            mid_low  = min(closes[-6:-2])
            start_p  = closes[-6]
            current  = closes[-1]
            drop_pct = (start_p - mid_low) / start_p * 100 if start_p > 0 else 0
            recov    = (current - mid_low) / mid_low * 100 if mid_low > 0 else 0
            if drop_pct > 5 and recov > 4:
                score = 15; pattern = "v_recovery"
                desc  = f"V-recovery after {drop_pct:.1f}% drop"

        return max(-25, min(25, score)), pattern, desc
    except Exception as e:
        print(f"  pattern_recognition error: {e}")
        return 0, "error", ""

# ── Feature 6: Earnings risk ──────────────────────────────────

def earnings_risk(api, ticker):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=2)
        news  = api.get_news(ticker,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=20)
        count = len(news) if news else 0
        words = ["earnings","eps","revenue","quarterly","guidance",
                 "results","beat","miss","q1","q2","q3","q4"]
        hits  = sum(1 for n in (news or [])
                    if any(w in n.headline.lower() for w in words))
        if hits >= 2:     return "high",    -20
        elif hits == 1 or count > 15: return "medium", -8
        else:             return "low",       0
    except:
        return "unknown", 0

# ── Feature 7: Market condition forecast ─────────────────────

def market_condition_forecast(api, regime):
    try:
        bars = safe_bars(api, "UVXY", days=10, limit=10)
        vix_move   = 0
        vix_rising = False; vix_falling = False
        if bars is not None and len(bars) >= 5:
            closes     = list(bars["close"])
            vix_move   = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] > 0 else 0
            vix_rising  = vix_move >  5
            vix_falling = vix_move < -5

        score = 0; condition = "neutral"
        if   regime == "trending_up"   and vix_falling: score = 15;  condition = "bullish"
        elif regime == "trending_up"   and not vix_rising: score = 8; condition = "mildly_bullish"
        elif regime == "trending_down" and vix_rising:  score = -15; condition = "bearish"
        elif regime == "trending_down" and not vix_falling: score = -8; condition = "mildly_bearish"
        elif regime == "ranging":  score = 2;  condition = "neutral"
        elif vix_rising:           score = -10; condition = "elevated_fear"
        elif vix_falling:          score = 5;  condition = "calming"

        return condition, score, round(vix_move, 1)
    except Exception as e:
        print(f"  market_condition_forecast error: {e}")
        return "unknown", 0, 0

# ── ATR-based stop/target calculator ─────────────────────────

def calculate_stops(api, ticker, entry_price, stop_pct=0.05, target_pct=0.10):
    """
    Calculates ATR-adjusted stop and target.
    Uses 1.5x ATR for stop (respects normal volatility).
    Uses 3x ATR for target (2:1 reward/risk minimum).
    Falls back to percentage if ATR not available.
    """
    bars = safe_bars(api, ticker, days=10, limit=10)
    if bars is not None and len(bars) >= 5:
        atr = calc_atr(bars, period=min(10, len(bars)-1))
        if atr > 0:
            stop   = round(entry_price - (atr * 1.5), 3)
            target = round(entry_price + (atr * 3.0), 3)
            # Ensure stop isn't tighter than 2% or wider than 10%
            stop   = max(stop,   entry_price * 0.90)
            stop   = min(stop,   entry_price * 0.98)
            target = max(target, entry_price * 1.05)
            return stop, target, round(atr, 4)

    # Fallback to fixed percentages
    return (
        round(entry_price * (1 - stop_pct),   3),
        round(entry_price * (1 + target_pct),  3),
        None
    )

# ── Master prediction ─────────────────────────────────────────

def predict_ticker(api, ticker, regime="ranging"):
    result = {
        "ticker":     ticker,
        "score":      0,
        "label":      "neutral",
        "confidence": "low",
        "components": {},
        "signals":    [],
        "tf_bias":    0,
        "tf_detail":  "",
        "timestamp":  datetime.now(NY).strftime("%I:%M %p ET"),
    }
    try:
        bars = safe_bars(api, ticker, days=20, limit=20)

        # Multi-timeframe first — this gates everything else
        tf_bias, tf_detail = multi_timeframe_analysis(api, ticker)

        sent_score,  sent_dir,   sent_daily = news_sentiment_trend(api, ticker)
        dir_score,   dir_conf,   dir_sigs   = price_direction_predictor(bars)
        vol_score,   vol_state,  vol_det    = volatility_forecast(bars)
        pat_score,   pat_name,   pat_desc   = pattern_recognition(bars)
        earn_risk,   earn_adj               = earnings_risk(api, ticker)
        mkt_cond,    mkt_score,  vix_move   = market_condition_forecast(api, regime)

        # Multi-timeframe bias adjustment:
        # If daily trend opposes prediction, dampen score significantly
        tf_multiplier = 1.0
        if tf_bias == 1 and dir_score < 0:
            tf_multiplier = 0.5   # daily up but intraday weak — dampen
        elif tf_bias == -1 and dir_score > 0:
            tf_multiplier = 0.5   # daily down but intraday strong — dampen
        elif tf_bias == 1:
            tf_multiplier = 1.2   # daily up confirms — boost
        elif tf_bias == -1:
            tf_multiplier = 1.2   # daily down confirms — boost (negative)

        total = (
            sent_score * 1.0 +
            dir_score  * 1.0 * tf_multiplier +
            vol_score  * 0.5 +
            pat_score  * 0.8 * tf_multiplier +
            earn_adj   * 1.0 +
            mkt_score  * 0.7 +
            tf_bias    * 10  # direct contribution from daily trend
        )
        total = max(-100, min(100, round(total, 1)))

        if   total >  50: label = "strong_buy"
        elif total >  25: label = "bullish"
        elif total >   0: label = "mildly_bullish"
        elif total > -25: label = "mildly_bearish"
        elif total > -50: label = "bearish"
        else:             label = "strong_avoid"

        confidence = ("high"   if dir_conf == "high"   and abs(pat_score) > 15 else
                      "medium" if dir_conf == "medium"  or abs(pat_score) > 8  else
                      "low")

        result.update({
            "score":      total,
            "label":      label,
            "confidence": confidence,
            "tf_bias":    tf_bias,
            "tf_detail":  tf_detail,
            "components": {
                "sentiment_trend": {
                    "score":     sent_score,
                    "direction": sent_dir,
                    "daily":     sent_daily,
                },
                "price_direction": {
                    "score":      dir_score,
                    "confidence": dir_conf,
                    "signals":    dir_sigs,
                },
                "volatility": {
                    "score":   vol_score,
                    "state":   vol_state,
                    "details": vol_det,
                },
                "pattern": {
                    "score":       pat_score,
                    "name":        pat_name,
                    "description": pat_desc,
                },
                "earnings_risk": {
                    "level":      earn_risk,
                    "adjustment": earn_adj,
                },
                "market_condition": {
                    "condition": mkt_cond,
                    "score":     mkt_score,
                    "vix_move":  vix_move,
                },
                "timeframe": {
                    "bias":   tf_bias,
                    "detail": tf_detail,
                },
            },
            "signals": dir_sigs,
        })
    except Exception as e:
        print(f"predict_ticker error {ticker}: {e}")

    lbl = result["label"]
    tf  = result["tf_detail"]
    print(f"  Pred {ticker}: {result['score']:+.0f} ({lbl}) tf={tf}")
    return result

def run_predictions(api, tickers, regime="ranging"):
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = predict_ticker(api, ticker, regime)
            time.sleep(0.5)
        except Exception as e:
            print(f"  run_predictions error {ticker}: {e}")
            results[ticker] = {
                "ticker": ticker, "score": 0, "label": "neutral",
                "confidence": "low", "tf_bias": 0, "tf_detail": ""
            }
    return results