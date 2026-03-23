import os, time, threading
from datetime import datetime
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# ================= CONFIG =================
API_KEY    = os.environ.get("APCA_API_KEY_ID")
SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

TICKERS   = ["AAPL", "TSLA", "NVDA", "MSFT"]
RISK_PCT  = 0.01
INTERVAL  = 60
STOP_LOSS = 0.02
TAKE_PROFIT = 0.04

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys.")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

price_history = {t: [] for t in TICKERS}
volume_history = {t: [] for t in TICKERS}
positions = {}
trade_log = []

# ================= INDICATORS =================

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = 0, 0
    for i in range(-period, 0):
        d = prices[i] - prices[i-1]
        if d > 0: gains += d
        else: losses += abs(d)
    if losses == 0: return 100
    return 100 - (100 / (1 + gains/losses))

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s) / len(s)

def calc_bollinger(prices, n=20):
    s = prices[-n:] if len(prices) >= n else prices
    mean = sum(s) / len(s)
    std = (sum((v - mean)**2 for v in s) / len(s)) ** 0.5
    return mean, mean + 2*std, mean - 2*std

def momentum(prices, n=5):
    if len(prices) < n: return 0
    return prices[-1] - prices[-n]

def volume_spike(volumes):
    if len(volumes) < 5: return False
    avg = sum(volumes[:-1]) / len(volumes[:-1])
    return volumes[-1] > avg * 1.5

# ================= MARKET REGIME =================

def market_regime(prices):
    if len(prices) < 50:
        return "neutral"
    ma50 = calc_ma(prices, 50)
    ma200 = calc_ma(prices, min(200, len(prices)))

    if ma50 > ma200 * 1.01:
        return "bull"
    elif ma200 > ma50 * 1.01:
        return "bear"
    else:
        return "sideways"

# ================= CANDLE PATTERNS =================

def bullish_engulfing(prev_open, prev_close, open_, close):
    return prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close

def bearish_engulfing(prev_open, prev_close, open_, close):
    return prev_close > prev_open and close < open_ and close < prev_open and open_ > prev_close

# ================= SIGNAL ENGINE =================

def get_score(ticker, price, bar):
    hist = price_history[ticker]
    vols = volume_history[ticker]

    if len(hist) < 20:
        return 0

    rsi = calc_rsi(hist)
    ma50 = calc_ma(hist, min(50, len(hist)))
    ma200 = calc_ma(hist, min(200, len(hist)))
    mean, upper, lower = calc_bollinger(hist)
    mom = momentum(hist)
    regime = market_regime(hist)

    score = 0

    # Trend
    if ma50 > ma200: score += 20
    else: score -= 20

    # RSI
    if rsi < 30: score += 15
    elif rsi > 70: score -= 15

    # Bollinger
    if price < lower: score += 10
    elif price > upper: score -= 10

    # Momentum
    if mom > 0: score += 10
    else: score -= 10

    # Volume
    if volume_spike(vols): score += 15

    # Regime adjustment
    if regime == "bull": score += 10
    elif regime == "bear": score -= 10

    return score

# ================= RISK =================

def position_size(price):
    acct = api.get_account()
    equity = float(acct.equity)
    return max(1, int((equity * RISK_PCT) / price))

# ================= EXECUTION =================

def execute(ticker, action, price):
    try:
        if action == "buy":
            qty = position_size(price)
            api.submit_order(symbol=ticker, qty=qty, side="buy", type="market", time_in_force="day")
            positions[ticker] = price
            trade_log.insert(0, {"type":"BUY","ticker":ticker,"qty":qty,"price":round(price,2)})

        elif action == "sell":
            pos = api.get_position(ticker)
            qty = int(pos.qty)
            api.submit_order(symbol=ticker, qty=qty, side="sell", type="market", time_in_force="day")
            entry = positions.get(ticker, price)
            pnl = round((price - entry) * qty, 2)
            positions.pop(ticker, None)
            trade_log.insert(0, {"type":"SELL","ticker":ticker,"qty":qty,"price":round(price,2),"pnl":pnl})
    except Exception as e:
        print("Order error:", e)

# ================= BOT LOOP =================

def bot_loop():
    while True:
        try:
            clock = api.get_clock()
            if not clock.is_open:
                time.sleep(60)
                continue

            state = {"tickers":{}, "market_status":"open"}

            for ticker in TICKERS:
                bar = api.get_latest_bar(ticker)
                price = float(bar.c)
                volume = float(bar.v)

                price_history[ticker].append(price)
                volume_history[ticker].append(volume)

                if len(price_history[ticker]) > 200:
                    price_history[ticker].pop(0)
                    volume_history[ticker].pop(0)

                score = get_score(ticker, price, bar)

                action = "hold"
                if score > 40:
                    action = "buy"
                elif score < -40:
                    action = "sell"

                # Risk exits
                if ticker in positions:
                    entry = positions[ticker]
                    if price <= entry * (1 - STOP_LOSS):
                        action = "sell"
                    elif price >= entry * (1 + TAKE_PROFIT):
                        action = "sell"

                if action != "hold":
                    execute(ticker, action, price)

                state["tickers"][ticker] = {
                    "price": round(price,2),
                    "score": score,
                    "action": action
                }

            state["trades"] = trade_log[:40]
            socketio.emit("state", state)

        except Exception as e:
            print("Loop error:", e)

        time.sleep(INTERVAL)

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/state")
def state_json():
    return jsonify({"trades": trade_log[:20]})

# ================= START =================

if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
