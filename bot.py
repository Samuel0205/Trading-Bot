import os, time, threading
from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from scanner import run_full_scan

API_KEY    = os.environ.get("APCA_API_KEY_ID")
SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys.")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

FALLBACK_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]
RISK_PCT         = 0.015
THRESHOLD        = 3
INTERVAL         = 60
ATR_STOP_MULT    = 1.5   # stop-loss = entry - (ATR * multiplier)
ATR_TARGET_MULT  = 3.0   # take-profit = entry + (ATR * multiplier)

price_history  = {}
volume_history = {}
trade_log      = []
scan_results   = {"today": [], "yesterday": [], "scanned_at": None}
active_tickers = list(FALLBACK_TICKERS)
open_positions = {}   # { ticker: { entry, stop, target, qty } }
market_regime  = "unknown"  # "trending_up", "trending_down", "ranging"

NY = pytz.timezone("America/New_York")

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
    return 100 - (100 / (1 + gains / losses))

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s) / len(s)

def calc_bollinger(prices, n=20):
    s = prices[-n:] if len(prices) >= n else prices
    mean = sum(s) / len(s)
    std  = (sum((v - mean)**2 for v in s) / len(s)) ** 0.5
    return mean, mean + 2*std, mean - 2*std

def calc_atr(highs, lows, closes, period=14):
    """Average True Range — measures volatility for stop sizing."""
    if len(closes) < 2:
        return closes[-1] * 0.02 if closes else 1.0
    trs = []
    for i in range(1, min(len(closes), period + 1)):
        hl  = highs[i]  - lows[i]
        hpc = abs(highs[i]  - closes[i-1])
        lpc = abs(lows[i]   - closes[i-1])
        trs.append(max(hl, hpc, lpc))
    return sum(trs) / len(trs) if trs else closes[-1] * 0.02

def calc_vwap(prices, volumes):
    """VWAP = sum(price * volume) / sum(volume) — intraday anchor."""
    if not prices or not volumes or sum(volumes) == 0:
        return prices[-1] if prices else 0
    return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)

def calc_macd(prices):
    """MACD line = EMA12 - EMA26, Signal = EMA9 of MACD."""
    def ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        k = 2 / (period + 1)
        val = sum(data[:period]) / period
        for p in data[period:]:
            val = p * k + val * (1 - k)
        return val
    if len(prices) < 26:
        return 0, 0
    macd_line = ema(prices, 12) - ema(prices, 26)
    return macd_line, macd_line  # signal needs history; simplified here

def get_signals(ticker, price):
    hist = price_history.get(ticker, [])
    vols = volume_history.get(ticker, [])

    if len(hist) < 5:
        return [{"name": n, "action": "hold", "signal": 50}
                for n in ["MA Crossover","RSI","Bollinger","VWAP","MACD","Mean Reversion"]]

    rsi              = calc_rsi(hist)
    ma50             = calc_ma(hist, min(50,  len(hist)))
    ma200            = calc_ma(hist, min(200, len(hist)))
    mean, upper, lower = calc_bollinger(hist)
    vwap             = calc_vwap(hist[-78:], vols[-78:])  # ~78 min = full session
    macd_line, _     = calc_macd(hist)
    z_score          = (price - mean) / max((upper - mean), 0.01)

    # Regime filter — only trade in direction of market trend
    regime_ok_buy  = market_regime in ("trending_up",  "ranging", "unknown")
    regime_ok_sell = market_regime in ("trending_down", "ranging", "unknown")

    def act(buy_cond, sell_cond, regime_buy=True, regime_sell=True):
        if buy_cond  and regime_ok_buy  and regime_buy:  return "buy"
        if sell_cond and regime_ok_sell and regime_sell: return "sell"
        return "hold"

    return [
        {"name": "MA Crossover",
         "action": act(ma50 > ma200*1.005, ma200 > ma50*1.005),
         "signal": min(95, 50 + (ma50-ma200)/max(ma200,1)*600)},
        {"name": "RSI",
         "action": act(rsi < 32, rsi > 68),
         "signal": 100 - rsi},
        {"name": "Bollinger",
         "action": act(price < lower*0.99, price > upper*1.01),
         "signal": max(5, min(95, 50 - z_score*40))},
        {"name": "VWAP",
         "action": act(price > vwap*1.001, price < vwap*0.999),
         "signal": min(95, max(5, 50 + (price-vwap)/max(vwap,1)*500))},
        {"name": "MACD",
         "action": act(macd_line > 0, macd_line < 0),
         "signal": min(95, max(5, 50 + macd_line * 10))},
        {"name": "Mean Reversion",
         "action": act(price < mean*0.96, price > mean*1.04),
         "signal": min(92, max(8, 50 + (mean-price)/max(mean,1)*250))},
    ]

# ── Market regime ─────────────────────────────────────────────

def update_market_regime():
    """Check SPY to determine if market is trending or ranging."""
    global market_regime
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=10)
        bars  = api.get_bars("SPY", "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=10).df
        if bars.empty or len(bars) < 5:
            return
        closes = list(bars["close"])
        ma5    = calc_ma(closes, 5)
        ma10   = calc_ma(closes, min(10, len(closes)))
        latest = closes[-1]

        if ma5 > ma10 * 1.005 and latest > ma5:
            market_regime = "trending_up"
        elif ma5 < ma10 * 0.995 and latest < ma5:
            market_regime = "trending_down"
        else:
            market_regime = "ranging"
        print(f"Market regime: {market_regime} (SPY ${latest:.2f})")
    except Exception as e:
        print(f"Regime error: {e}")

# ── ATR-based stop/target management ─────────────────────────

def fetch_atr(ticker):
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=5)
        bars  = api.get_bars(ticker, "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=5).df
        if bars.empty:
            return None
        return calc_atr(
            list(bars["high"]),
            list(bars["low"]),
            list(bars["close"])
        )
    except:
        return None

def check_stops(ticker, price):
    """Check if any open position has hit its stop-loss or take-profit."""
    pos = open_positions.get(ticker)
    if not pos:
        return
    if price <= pos["stop"]:
        print(f"STOP-LOSS hit: {ticker} @ ${price:.2f} (stop ${pos['stop']:.2f})")
        force_sell(ticker, price, reason="stop_loss")
    elif price >= pos["target"]:
        print(f"TAKE-PROFIT hit: {ticker} @ ${price:.2f} (target ${pos['target']:.2f})")
        force_sell(ticker, price, reason="take_profit")

def force_sell(ticker, price, reason="stop_loss"):
    try:
        pos = api.get_position(ticker)
        qty = int(pos.qty)
        if qty <= 0:
            open_positions.pop(ticker, None)
            return
        pnl = round((price - float(pos.avg_entry_price)) * qty, 2)
        api.submit_order(symbol=ticker, qty=qty, side="sell",
                         type="market", time_in_force="day")
        trade_log.insert(0, {
            "type": "SELL", "ticker": ticker, "qty": qty,
            "price": round(price, 2), "pnl": pnl, "reason": reason
        })
        open_positions.pop(ticker, None)
        print(f"FORCE SELL {qty}x {ticker} @ ${price:.2f} | {reason} | PnL ${pnl}")
    except Exception as e:
        print(f"Force sell error {ticker}: {e}")

# ── Helpers ───────────────────────────────────────────────────

def is_market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def get_account_state():
    acct = api.get_account()
    return {
        "portfolio": round(float(acct.equity), 2),
        "cash":      round(float(acct.cash), 2),
        "pnl":       round(float(acct.equity) - float(acct.last_equity), 2),
        "regime":    market_regime,
    }

def position_size(price, atr):
    """Kelly-inspired sizing: risk exactly RISK_PCT of portfolio per trade."""
    acct      = api.get_account()
    equity    = float(acct.equity)
    risk_amt  = equity * RISK_PCT
    # Use ATR as the risk-per-share (stop distance)
    stop_dist = atr * ATR_STOP_MULT if atr else price * 0.02
    return max(1, int(risk_amt / stop_dist))

def close_all_positions_eod():
    """Force-close all positions 15 min before market close."""
    try:
        positions = api.list_positions()
        for pos in positions:
            ticker = pos.symbol
            price  = float(pos.current_price)
            force_sell(ticker, price, reason="eod_close")
        print("EOD: all positions closed")
    except Exception as e:
        print(f"EOD close error: {e}")

# ── Scanner loop ──────────────────────────────────────────────

def scanner_loop():
    global scan_results
    last_scan_date = None
    while True:
        now   = datetime.now(NY)
        today = now.date()
        if now.weekday() < 5 and now.hour >= 9 and last_scan_date != today:
            print("Starting daily scan...")
            try:
                update_market_regime()
                scan_results   = run_full_scan(api)
                last_scan_date = today
                if scan_results["today"]:
                    new_tickers = [s["ticker"] for s in scan_results["today"]]
                    for t in new_tickers:
                        if t not in price_history:
                            price_history[t]  = []
                            volume_history[t] = []
                    active_tickers.clear()
                    active_tickers.extend(new_tickers)
                    print(f"Bot now trading: {active_tickers}")
                socketio.emit("scan", scan_results)
            except Exception as e:
                print(f"Scanner error: {e}")
        time.sleep(300)

# ── On connect ────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("Client connected")
    try:
        market_open = is_market_open()
        state = {
            "tickers":       {},
            "account":       get_account_state(),
            "trades":        trade_log[:40],
            "market_status": "open" if market_open else "closed",
        }
        if market_open:
            for ticker in list(active_tickers):
                try:
                    bar   = api.get_latest_bar(ticker)
                    price = float(bar.c)
                    price_history.setdefault(ticker, []).append(price)
                    volume_history.setdefault(ticker, []).append(float(bar.v))
                    sigs  = get_signals(ticker, price)
                    buys  = sum(1 for s in sigs if s["action"] == "buy")
                    sells = sum(1 for s in sigs if s["action"] == "sell")
                    action = "buy" if buys >= THRESHOLD else "sell" if sells >= THRESHOLD else "hold"
                    pos = open_positions.get(ticker)
                    state["tickers"][ticker] = {
                        "price": round(price, 2), "signals": sigs,
                        "action": action, "buys": buys, "sells": sells,
                        "stop":   round(pos["stop"],   2) if pos else None,
                        "target": round(pos["target"], 2) if pos else None,
                    }
                except Exception as e:
                    print(f"on_connect {ticker}: {e}")
        socketio.emit("state", state)
        if scan_results["scanned_at"]:
            socketio.emit("scan", scan_results)
    except Exception as e:
        print(f"on_connect error: {e}")

# ── Trade execution ───────────────────────────────────────────

def execute(ticker, action, price):
    try:
        if action == "buy" and ticker not in open_positions:
            atr = fetch_atr(ticker)
            qty = position_size(price, atr)
            stop   = price - (atr * ATR_STOP_MULT)   if atr else price * 0.98
            target = price + (atr * ATR_TARGET_MULT)  if atr else price * 1.06
            api.submit_order(symbol=ticker, qty=qty, side="buy",
                             type="market", time_in_force="day")
            open_positions[ticker] = {"entry": price, "stop": stop, "target": target, "qty": qty}
            trade_log.insert(0, {
                "type": "BUY", "ticker": ticker, "qty": qty,
                "price": round(price, 2), "pnl": None,
                "stop": round(stop, 2), "target": round(target, 2)
            })
            print(f"BUY {qty}x {ticker} @ ${price:.2f} | stop ${stop:.2f} | target ${target:.2f}")

        elif action == "sell" and ticker in open_positions:
            force_sell(ticker, price, reason="signal")

    except Exception as e:
        print(f"Order error {ticker}: {e}")

# ── Main bot loop ─────────────────────────────────────────────

def bot_loop():
    last_regime_update = None
    while True:
        try:
            market_open = is_market_open()
            now = datetime.now(NY)

            if not market_open:
                socketio.emit("state", {
                    "tickers": {}, "account": get_account_state(),
                    "trades": trade_log[:40], "market_status": "closed"
                })
                time.sleep(60)
                continue

            # Update regime every 30 min during market hours
            if not last_regime_update or (now - last_regime_update).seconds > 1800:
                update_market_regime()
                last_regime_update = now

            # EOD close — 15 min before close (3:45 PM ET)
            if now.hour == 15 and now.minute >= 45:
                close_all_positions_eod()
                time.sleep(900)
                continue

            state = {"tickers": {}, "account": {}, "market_status": "open", "regime": market_regime}

            for ticker in list(active_tickers):
                try:
                    bar   = api.get_latest_bar(ticker)
                    price = float(bar.c)
                    vol   = float(bar.v)

                    price_history.setdefault(ticker,  []).append(price)
                    volume_history.setdefault(ticker, []).append(vol)
                    if len(price_history[ticker])  > 200: price_history[ticker].pop(0)
                    if len(volume_history[ticker]) > 200: volume_history[ticker].pop(0)

                    # Check stops before signal logic
                    check_stops(ticker, price)

                    sigs  = get_signals(ticker, price)
                    buys  = sum(1 for s in sigs if s["action"] == "buy")
                    sells = sum(1 for s in sigs if s["action"] == "sell")
                    action = "buy" if buys >= THRESHOLD else "sell" if sells >= THRESHOLD else "hold"

                    if action != "hold":
                        execute(ticker, action, price)

                    pos = open_positions.get(ticker)
                    state["tickers"][ticker] = {
                        "price":  round(price, 2), "signals": sigs,
                        "action": action, "buys": buys, "sells": sells,
                        "stop":   round(pos["stop"],   2) if pos else None,
                        "target": round(pos["target"], 2) if pos else None,
                    }
                    print(f"  {ticker}: ${price:.2f} | {action} | regime:{market_regime}")
                except Exception as e:
                    print(f"  {ticker} error: {e}")

            state["account"] = get_account_state()
            state["trades"]  = trade_log[:40]
            socketio.emit("state", state)

        except Exception as e:
            print(f"Loop error: {e}")

        time.sleep(INTERVAL)

# ── Flask routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/scanner")
def scanner_page():
    return render_template("scanner.html")

@app.route("/predictions")
def predictions_page():
    return render_template("predictions.html")

@app.route("/state")
def state_json():
    return jsonify({"trades": trade_log[:20]})

@app.route("/scan")
def scan_json():
    return jsonify(scan_results)

if __name__ == "__main__":
    threading.Thread(target=bot_loop,     daemon=True).start()
    threading.Thread(target=scanner_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
