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
RISK_PCT  = 0.015
THRESHOLD = 3
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

# ================= ACCOUNT =================
def get_account_state():
    acct = api.get_account()
    return {
        "portfolio": round(float(acct.equity), 2),
        "cash": round(float(acct.cash), 2),
        "pnl": round(float(acct.equity) - float(acct.last_equity), 2),
    }

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

# ================= SIGNAL ENGINE (UI SAFE) =================
def get_signals(ticker, price):
    hist = price_history[ticker]
    vols = volume_history[ticker]

    if len(hist) < 5:
        return [{"name": n, "action": "hold", "signal": 50}
                for n in ["MA Crossover","RSI","Bollinger","Mean Reversion"]]

    rsi = calc_rsi(hist)
    ma50 = calc_ma(hist, min(50, len(hist)))
    ma200 = calc_ma(hist, min(200, len(hist)))
    mean, upper, lower = calc_bollinger(hist)
    mom = momentum(hist)

    signals = []

    # MA Crossover
    signals.append({
        "name": "MA Crossover",
        "action": "buy" if ma50 > ma200 else "sell",
        "signal": 70
    })

    # RSI
    signals.append({
        "name": "RSI",
        "action": "buy" if rsi < 32 else "sell" if rsi > 68 else "hold",
        "signal": 100 - rsi
    })

    # Bollinger
    signals.append({
        "name": "Bollinger",
        "action": "buy" if price < lower else "sell" if price > upper else "hold",
        "signal": 60
    })

    # Mean Reversion
    signals.append({
        "name": "Mean Reversion",
        "action": "buy" if price < mean else "sell" if price > mean else "hold",
        "signal": 60
    })

    # Momentum boost
    if mom > 0:
        signals.append({"name":"Momentum","action":"buy","signal":65})
    else:
        signals.append({"name":"Momentum","action":"sell","signal":65})

    # Volume boost
    if volume_spike(vols):
        signals.append({"name":"Volume Spike","action":"buy","signal":80})

    return signals

# ================= EXECUTION =================
def position_size(price):
    acct = api.get_account()
    return max(1, int(float(acct.equity) * RISK_PCT / price))

def execute(ticker, action, price):
    try:
        if action == "buy":
            qty = position_size(price)
            api.submit_order(symbol=ticker, qty=qty, side="buy",
                             type="market", time_in_force="day")
            positions[ticker] = price
            trade_log.insert(0, {"type":"BUY","ticker":ticker,"qty":qty,"price":round(price,2)})

        elif action == "sell":
            pos = api.get_position(ticker)
            qty = max(1, int(int(pos.qty) / 2))
            pnl = round((price - float(pos.avg_entry_price)) * qty, 2)

            api.submit_order(symbol=ticker, qty=qty, side="sell",
                             type="market", time_in_force="day")

            trade_log.insert(0, {"type":"SELL","ticker":ticker,"qty":qty,"price":round(price,2),"pnl":pnl})

    except Exception as e:
        print("Order error:", e)

# ================= SOCKET CONNECT =================
@socketio.on("connect")
def on_connect():
    print("Client connected")
    state = {
        "tickers": {},
        "account": get_account_state(),
        "trades": trade_log[:40],
        "market_status": "open"
    }
    socketio.emit("state", state)

# ================= BOT LOOP =================
def bot_loop():
    while True:
        try:
            clock = api.get_clock()

            if not clock.is_open:
                socketio.emit("state", {
                    "tickers": {},
                    "account": get_account_state(),
                    "trades": trade_log[:40],
                    "market_status": "closed"
                })
                time.sleep(60)
                continue

            state = {
                "tickers": {},
                "account": {},
                "trades": trade_log[:40],
                "market_status": "open"
            }

            for ticker in TICKERS:
                bar = api.get_latest_bar(ticker)
                price = float(bar.c)
                volume = float(bar.v)

                price_history[ticker].append(price)
                volume_history[ticker].append(volume)

                if len(price_history[ticker]) > 200:
                    price_history[ticker].pop(0)
                    volume_history[ticker].pop(0)

                sigs = get_signals(ticker, price)

                buys  = sum(1 for s in sigs if s["action"] == "buy")
                sells = sum(1 for s in sigs if s["action"] == "sell")

                action = "buy" if buys >= THRESHOLD else "sell" if sells >= THRESHOLD else "hold"

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
                    "price": round(price, 2),
                    "signals": sigs,
                    "action": action,
                    "buys": buys,
                    "sells": sells
                }

            state["account"] = get_account_state()
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
