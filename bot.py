import os, time, json, threading
from datetime import datetime
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

API_KEY    = os.environ.get("AKKV2ZK5ZAHGNPPY4JT7SLWPF2")
SECRET_KEY = os.environ.get("A6R4jsRgJiLoFGJR3ZpzbThGktZsVkkY4V4xpmXHvqzR")
BASE_URL   = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

TICKERS    = ["AAPL", "TSLA", "NVDA", "MSFT"]
RISK_PCT   = 0.015   # 1.5% of portfolio per trade
THRESHOLD  = 3       # bots that must agree to trade
INTERVAL   = 60      # seconds between cycles (1 min)

price_history = {t: [] for t in TICKERS}
trade_log     = []

# ── Indicators ────────────────────────────────────────────────

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = 0, 0
    for i in range(-period, 0):
        d = prices[i] - prices[i-1]
        if d > 0: gains += d
        else:     losses += abs(d)
    if losses == 0: return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s) / len(s)

def calc_bollinger(prices, n=20):
    s = prices[-n:] if len(prices) >= n else prices
    mean = sum(s) / len(s)
    std  = (sum((v - mean)**2 for v in s) / len(s)) ** 0.5
    return mean, mean + 2*std, mean - 2*std

def get_signals(ticker, price):
    hist = price_history[ticker]
    if len(hist) < 5:
        return [{"action": "hold", "signal": 50}] * 4

    rsi         = calc_rsi(hist)
    ma50        = calc_ma(hist, min(50,  len(hist)))
    ma200       = calc_ma(hist, min(200, len(hist)))
    mean, upper, lower = calc_bollinger(hist)
    z_score     = (price - mean) / max((upper - mean), 0.01)

    return [
        # MA Crossover
        {"name": "MA Crossover",
         "action": "buy" if ma50 > ma200*1.005 else "sell" if ma200 > ma50*1.005 else "hold",
         "signal": min(95, 50 + (ma50-ma200)/max(ma200,1)*600)},
        # RSI
        {"name": "RSI",
         "action": "buy" if rsi < 32 else "sell" if rsi > 68 else "hold",
         "signal": 100 - rsi},
        # Bollinger Bands
        {"name": "Bollinger",
         "action": "buy" if price < lower*0.99 else "sell" if price > upper*1.01 else "hold",
         "signal": max(5, min(95, 50 - z_score*40))},
        # Mean Reversion
        {"name": "Mean Reversion",
         "action": "buy" if price < mean*0.96 else "sell" if price > mean*1.04 else "hold",
         "signal": min(92, max(8, 50 + (mean-price)/max(mean,1)*250))},
    ]

# ── Market hours check ────────────────────────────────────────

def is_market_open():
    ny  = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    if now.weekday() >= 5:
        return False
    return 9 <= now.hour < 16

# ── Position sizing ───────────────────────────────────────────

def position_size(price):
    account    = api.get_account()
    portfolio  = float(account.portfolio_value)
    risk_amt   = portfolio * RISK_PCT
    return max(1, int(risk_amt / price))

# ── Trade execution ───────────────────────────────────────────

def execute(ticker, action, price):
    try:
        if action == "buy":
            qty = position_size(price)
            api.submit_order(
                symbol=ticker, qty=qty,
                side="buy", type="market",
                time_in_force="day"
            )
            trade_log.insert(0, {"type":"BUY","ticker":ticker,"qty":qty,"price":round(price,2),"pnl":None})
            print(f"BUY  {qty}x {ticker} @ ${price:.2f}")

        elif action == "sell":
            try:
                pos = api.get_position(ticker)
                qty = max(1, int(int(pos.qty) / 2))
                avg = float(pos.avg_entry_price)
                pnl = round((price - avg) * qty, 2)
                api.submit_order(
                    symbol=ticker, qty=qty,
                    side="sell", type="market",
                    time_in_force="day"
                )
                trade_log.insert(0, {"type":"SELL","ticker":ticker,"qty":qty,"price":round(price,2),"pnl":pnl})
                print(f"SELL {qty}x {ticker} @ ${price:.2f}  PnL ${pnl}")
            except:
                pass  # no position to sell
    except Exception as e:
        print(f"Order error: {e}")

# ── Main bot loop ─────────────────────────────────────────────

def bot_loop():
    while True:
        if not is_market_open():
            print("Market closed — sleeping 5 min")
            time.sleep(300)
            continue

        state = {"tickers": {}, "account": {}}

        for ticker in TICKERS:
            try:
                bar   = api.get_latest_bar(ticker)
                price = float(bar.c)
                price_history[ticker].append(price)
                if len(price_history[ticker]) > 200:
                    price_history[ticker].pop(0)

                sigs  = get_signals(ticker, price)
                buys  = sum(1 for s in sigs if s["action"] == "buy")
                sells = sum(1 for s in sigs if s["action"] == "sell")

                action = "hold"
                if buys  >= THRESHOLD: action = "buy"
                elif sells >= THRESHOLD: action = "sell"

                if action != "hold":
                    execute(ticker, action, price)

                state["tickers"][ticker] = {
                    "price":   round(price, 2),
                    "signals": sigs,
                    "action":  action,
                    "buys":    buys,
                    "sells":   sells,
                }
            except Exception as e:
                print(f"{ticker} error: {e}")

        try:
            acct = api.get_account()
            state["account"] = {
                "portfolio": round(float(acct.portfolio_value), 2),
                "cash":      round(float(acct.cash), 2),
                "pnl":       round(float(acct.portfolio_value) - 10000, 2),
            }
        except Exception as e:
            print(f"Account error: {e}")

        state["trades"] = trade_log[:40]
        socketio.emit("state", state)
        time.sleep(INTERVAL)

# ── Flask routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/state")
def state_json():
    return jsonify({"trades": trade_log[:20]})

if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
