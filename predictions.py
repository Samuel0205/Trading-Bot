"""
predictions.py — Multi-factor prediction engine

Features:
  - 3-day news sentiment trend (FinBERT via HuggingFace API or keyword fallback)
  - Next-day price direction (momentum trajectory + acceleration)
  - Volatility forecast (ATR expansion/contraction)
  - Pattern recognition (channels, breakouts, V-recovery)
  - Earnings risk flag
  - Market condition forecast (regime + fear proxy via UVXY)
  - Multi-timeframe confirmation (daily trend vs intraday)

Output: prediction_score (-100 to +100) per ticker
  > 30  = bullish  → bot gets confidence boost + larger size
  < -30 = bearish  → bot skips trade
  0±30  = neutral  → normal behavior
"""

import time, os
from datetime import datetime, timedelta
import pytz
from finbert_client import finbert_score, keyword_score as _fb_kw, calls_remaining

NY = pytz.timezone("America/New_York")

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
    bars = safe_bars(api, ticker, timeframe="1Day", days=15, limit=15)
    if bars is None or len(bars) < 5:
        return 0, "insufficient_data"

    closes = list(bars["close"])
    ma5    = calc_ma(closes, min(5,  len(closes)))
    ma10   = calc_ma(closes, min(10, len(closes)))
    latest = closes[-1]
    highs  = list(bars["high"])
    lows   = list(bars["low"])
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
    # Fetch headlines for the past 3 days separately so we can measure the trend.
    # FinBERT budget: one call batches ALL days' headlines together, preserving the
    # per-day breakdown via keyword scores scaled to FinBERT magnitude.
    day_headlines = []
    for days_ago in [3, 2, 1]:
        try:
            end   = datetime.now(pytz.utc) - timedelta(days=days_ago - 1)
            start = end - timedelta(days=1)
            news  = api.get_news(ticker,
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=5)
            day_headlines.append([n.headline for n in news] if news else [])
        except:
            day_headlines.append([])

    # Per-day keyword scores (always computed — used as trend backbone)
    kw_scores = [sum(_fb_kw(h) for h in day) for day in day_headlines]

    # One FinBERT call: batch all days' top headlines together.
    # Scale keyword per-day scores by FinBERT's total for better calibration.
    all_h = [h for day in day_headlines for h in day[:3]]
    if all_h:
        fb_total, method = finbert_score(all_h)
        kw_total = sum(kw_scores)
        if method == "finbert" and kw_total != 0:
            scale = fb_total / kw_total
            daily_scores = [round(s * scale, 3) for s in kw_scores]
        elif method == "finbert":
            # All keyword scores were 0 — split FinBERT total evenly
            daily_scores = [round(fb_total / 3, 3)] * 3
        else:
            daily_scores = kw_scores
    else:
        daily_scores = [0, 0, 0]
        method = "no_news"

    if len(daily_scores) < 2:
        return 0, "neutral", daily_scores

    trend      = daily_scores[-1] - daily_scores[0]
    normalized = max(-30, min(30, trend * 10))
    direction  = "improving" if trend > 0.1 else "worsening" if trend < -0.1 else "neutral"
    return round(normalized, 2), direction, daily_scores

# ── Feature 3: Price direction predictor ─────────────────────

def price_direction_predictor(bars):
    if bars is None or len(bars) < 5:
        return 0, "low", []

    closes  = list(bars["close"])
    volumes = list(bars["volume"])
    score   = 0
    signals = []

    if len(closes) >= 4:
        mom = (closes[-1] - closes[-4]) / closes[-4] * 100 if closes[-4] > 0 else 0
        if   mom >  3: score += 10; signals.append(f"3d momentum +{mom:.1f}%")
        elif mom >  1: score +=  5; signals.append(f"3d momentum +{mom:.1f}%")
        elif mom < -3: score -= 10; signals.append(f"3d momentum {mom:.1f}%")
        elif mom < -1: score -=  5; signals.append(f"3d momentum {mom:.1f}%")

    if len(closes) >= 6:
        recent = closes[-1] - closes[-3]
        prior  = closes[-3] - closes[-5]
        if recent > 0 and recent > prior:
            score += 8; signals.append("momentum accelerating ↑")
        elif recent < 0 and recent < prior:
            score -= 8; signals.append("momentum accelerating ↓")
        elif recent > 0 and recent < prior:
            score -= 3; signals.append("momentum decelerating")

    if len(volumes) >= 3:
        avg_vol   = sum(volumes[-10:]) / min(len(volumes), 10)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        last_move = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        if last_move > 0 and vol_ratio > 1.5:
            score += 8; signals.append(f"volume confirmation {vol_ratio:.1f}x")
        elif last_move < 0 and vol_ratio > 1.5:
            score -= 8; signals.append(f"volume selling {vol_ratio:.1f}x")
        elif last_move > 0 and vol_ratio < 0.7:
            score -= 3; signals.append("weak volume on up day")

    if len(closes) >= 10:
        ma5  = calc_ma(closes, 5)
        ma10 = calc_ma(closes, 10)
        p    = closes[-1]
        if p > ma5 > ma10:
            score += 7; signals.append("price > MA5 > MA10")
        elif p < ma5 < ma10:
            score -= 7; signals.append("price < MA5 < MA10")

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
    if bars is None or len(bars) < 6:
        return 0, "unknown", {}
    try:
        mid        = len(bars) // 2
        atr_prior  = calc_atr(bars.iloc[:mid])
        atr_recent = calc_atr(bars.iloc[mid:])
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
                score = 20; pattern = "uptrend";    desc = "Higher highs and higher lows"
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
        if hits >= 2:                  return "high",    -20
        elif hits == 1 or count > 15:  return "medium",  -8
        else:                          return "low",       0
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
        if   regime == "trending_up"   and vix_falling:      score = 15;  condition = "bullish"
        elif regime == "trending_up"   and not vix_rising:   score = 8;   condition = "mildly_bullish"
        elif regime == "trending_down" and vix_rising:       score = -15; condition = "bearish"
        elif regime == "trending_down" and not vix_falling:  score = -8;  condition = "mildly_bearish"
        elif regime == "ranging":                             score = 2;   condition = "neutral"
        elif vix_rising:                                      score = -10; condition = "elevated_fear"
        elif vix_falling:                                     score = 5;   condition = "calming"

        return condition, score, round(vix_move, 1)
    except Exception as e:
        print(f"  market_condition_forecast error: {e}")
        return "unknown", 0, 0

# ── ATR-based stop/target calculator ─────────────────────────

def calculate_stops(api, ticker, entry_price, stop_pct=0.05, target_pct=0.10):
    """
    Intraday stop/target with a fixed 2:1 reward/risk.

    The stop distance starts from 1.5×(daily ATR) but is CLAMPED to a reachable
    intraday band [1.5%, 4%] of price, and the target is always exactly 2× that
    stop distance. This is deliberate:

    - The old version set stop = entry-1.5×ATR (a *daily* ATR ≈ 8-10% for a meme
      stock) then clamped it to [2%,10%]. A ~10% stop and a 3×ATR (~28%) target
      almost never trigger within one session, so every position rode to the 3:28pm
      EOD market dump — the ATR "risk management" never actually engaged intraday.
    - It also disagreed with position_size(), which sized shares as if the stop sat
      at exactly 1.5×ATR, so realized risk ≠ RISK_PER_TRADE.

    Returning a tight, reachable stop and a consistent 2× target means exits fire
    intraday and the payoff is symmetric-positive (risk 1R, target 2R). Callers
    size the position off the ACTUAL stop distance (entry - stop) so $ risk is real.
    """
    bars = safe_bars(api, ticker, days=10, limit=10)
    atr  = None
    if bars is not None and len(bars) >= 5:
        a = calc_atr(bars, period=min(10, len(bars)-1))
        if a and a > 0:
            atr = a
    raw_dist  = (atr * 1.5) if atr else (entry_price * stop_pct)
    # Clamp the stop distance to a reachable intraday band: 1.5%–4% of price.
    stop_dist = min(max(raw_dist, entry_price * 0.015), entry_price * 0.04)
    stop      = round(entry_price - stop_dist, 3)
    target    = round(entry_price + stop_dist * 2.0, 3)   # fixed 2:1 reward/risk
    return stop, target, (round(atr, 4) if atr else None)

# ── Master prediction for one ticker ─────────────────────────

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
        "fetched_at": time.time(),
        "timestamp":  datetime.now(NY).strftime("%I:%M %p ET"),
    }
    try:
        bars = safe_bars(api, ticker, days=20, limit=20)

        tf_bias, tf_detail             = multi_timeframe_analysis(api, ticker)
        sent_score, sent_dir, sent_daily = news_sentiment_trend(api, ticker)
        dir_score,  dir_conf, dir_sigs   = price_direction_predictor(bars)
        vol_score,  vol_state, vol_det   = volatility_forecast(bars)
        pat_score,  pat_name,  pat_desc  = pattern_recognition(bars)
        earn_risk_lv, earn_adj           = earnings_risk(api, ticker)
        mkt_cond,   mkt_score, vix_move  = market_condition_forecast(api, regime)

        if   tf_bias == 1 and dir_score < 0:  tf_multiplier = 0.5
        elif tf_bias == -1 and dir_score > 0: tf_multiplier = 0.5
        elif tf_bias == 1:                    tf_multiplier = 1.2
        elif tf_bias == -1:                   tf_multiplier = 1.2
        else:                                 tf_multiplier = 1.0

        # Congressional STOCK Act activity — lagging (30-45 day disclosure delay)
        # but carries meaningful confirmation value.
        pol_raw = 0
        try:
            from politician_tracker import get_ticker_scores
            pol_raw = get_ticker_scores().get(ticker, 0)
        except Exception:
            pass
        pol_pred_score = round(min(15, pol_raw * 0.6), 1)

        # Corporate insider Form 4 purchases — faster disclosure (2 business days),
        # high-conviction signal when executives buy their own stock.
        insider_raw = 0
        try:
            from insider_tracker import get_insider_scores
            insider_raw = get_insider_scores().get(ticker, 0)
        except Exception:
            pass
        insider_pred_score = round(min(12, insider_raw * 0.6), 1)

        # StockTwits crowd sentiment + short squeeze potential — directly
        # relevant for the meme/momentum universe this bot targets.
        social_score = 0.0
        short_score  = 0.0
        try:
            from social_sentiment import get_social_score, get_short_score
            price_trend   = 1 if (bars is not None and len(bars) >= 2 and
                                  float(bars.iloc[-1]["close"]) > float(bars.iloc[-2]["close"])) else -1
            social_score  = get_social_score(ticker)
            short_score   = get_short_score(ticker, price_trend=price_trend)
        except Exception:
            pass
        # Social: -10 to +10 → scale to -8 to +8 in prediction space
        social_pred_score = round(max(-8, min(8, social_score * 0.8)), 1)
        # Short squeeze: 0-15 → 0-8 (only positive; squeezes are bullish)
        short_pred_score  = round(min(8, short_score * 0.55), 1)

        # SEC EDGAR + Yahoo Finance catalyst — fundamental driver behind price move
        catalyst_raw      = 0
        catalyst_headline = ""
        try:
            from catalyst_tracker import get_catalyst_score as _get_cat
            cat               = _get_cat(ticker)
            catalyst_raw      = cat.get("score", 0)
            catalyst_headline = cat.get("headline", "")
        except Exception:
            pass
        # Scale: catalyst raw (-20 to +20) → prediction space (-15 to +15)
        catalyst_pred_score = round(max(-15, min(15, catalyst_raw * 0.75)), 1)

        total = (
            sent_score          * 1.0 +
            dir_score           * 1.0 * tf_multiplier +
            vol_score           * 0.5 +
            pat_score           * 0.8 * tf_multiplier +
            earn_adj            * 1.0 +
            mkt_score           * 0.7 +
            tf_bias             * 10  +
            pol_pred_score      * 1.0 +
            insider_pred_score  * 1.2 +   # slightly higher weight: faster + more direct signal
            social_pred_score   * 0.8 +   # crowd can be noisy; moderate weight
            short_pred_score    * 0.9 +   # squeeze potential is directional but timing-dependent
            catalyst_pred_score * 1.5     # strong weight: real fundamental backing
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
            "fetched_at": time.time(),
            "components": {
                "sentiment_trend":    {"score": sent_score,         "direction": sent_dir, "daily": sent_daily},
                "price_direction":    {"score": dir_score,          "confidence": dir_conf, "signals": dir_sigs},
                "volatility":         {"score": vol_score,          "state": vol_state, "details": vol_det},
                "pattern":            {"score": pat_score,          "name": pat_name, "description": pat_desc},
                "earnings_risk":      {"level": earn_risk_lv,       "adjustment": earn_adj},
                "market_condition":   {"condition": mkt_cond,       "score": mkt_score, "vix_move": vix_move},
                "timeframe":          {"bias": tf_bias,             "detail": tf_detail},
                "politician_activity":{"score": pol_pred_score,     "raw_score": pol_raw},
                "insider_buying":     {"score": insider_pred_score, "raw_score": insider_raw},
                "social_sentiment":   {"score": social_pred_score,  "raw_score": social_score},
                "short_squeeze":      {"score": short_pred_score,   "raw_score": short_score},
                "catalyst":           {"score": catalyst_pred_score, "raw_score": catalyst_raw,
                                       "headline": catalyst_headline},
            },
            "signals": dir_sigs,
        })
    except Exception as e:
        print(f"predict_ticker error {ticker}: {e}")

    print(f"  Pred {ticker}: {result['score']:+.0f} ({result['label']}) tf={result['tf_detail']}")
    return result

# ── Batch runner — required by app.py ────────────────────────

def run_predictions(api, tickers, market_regime="ranging"):
    """
    Runs predictions for all active tickers.
    Called by prediction_loop() and /predictions/manual in app.py.
    Returns: { ticker: prediction_dict, ... }
    """
    results = {}
    print(f"=== run_predictions start: {tickers} | regime={market_regime} ===")
    for ticker in tickers:
        try:
            results[ticker] = predict_ticker(api, ticker, market_regime)
            time.sleep(0.4)
        except Exception as e:
            print(f"  run_predictions failed {ticker}: {e}")
            results[ticker] = {
                "ticker": ticker, "score": 0, "label": "neutral",
                "confidence": "low", "components": {}, "signals": [],
                "tf_bias": 0, "tf_detail": str(e),
                "timestamp": datetime.now(NY).strftime("%I:%M %p ET"),
                "fetched_at": time.time(),   # required for staleness gate in bot.py
            }
    print(f"=== run_predictions done: {len(results)} results ===")
    return results
