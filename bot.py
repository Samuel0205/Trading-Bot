import os, time, threading
from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from scanner import run_full_scan, build_universe

API_KEY    = os.environ.get("APCA_API_KEY_ID")
SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys.")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Config ────────────────────────────────────────────────────
MAX_ACCOUNT         = 20.00
MAX_TRADE_PCT       = 0.50
STOP_LOSS_PCT       = 0.05
TAKE_PROFIT_PCT     = 0.10
THRESHOLD           = 2
INTERVAL            = 15
MIN_PRICE           = 0.50
MIN_VOLUME          = 100_000
MIN_GRADE = ["A", "B", "C", "D"]

COOLDOWN_STOP       = 600
COOLDOWN_PROFIT     = 180
COOLDOWN_SIGNAL     = 300

TRADING_START       = (8, 45)
TRADING_END         = (14, 0)
SCAN_HOURS          = [9, 11, 13]

FALLBACK_TICKERS    = ["SIRI", "TELL", "CLOV", "NKLA", "MVIS"]

price_history  = {}
volume_history = {}
trade_log      = []
scan_results   = {"today": [], "yesterday": [], "scanned_at": None}
active_tickers = list(FALLBACK_TICKERS)
open_positions = {}
market_regime  = "unknown"
cooldowns      = {}
ticker_grades  = {}

NY = pytz.timezone("America/New_York")

# ── Helpers ───────────────────────────────────────────────────

def get_account():
    try:
        return api.get_account()
    except Exception as e:
        print(f"get_account error: {e}")
        return None

def get_account_size():
    acct = get_account()
    actual = float(acct.equity) if acct else MAX_ACCOUNT
    return min(actual, MAX_ACCOUNT)

def get_available_cash():
    acct = get_account()
    if not acct:
        return 0
    return min(float(acct.cash), float(acct.buying_power))

def get_account_state():
    try:
        acct = get_account()
        if not acct:
            return {"portfolio": 0, "cash": 0, "pnl": 0, "regime": market_regime}
        return {
            "portfolio": round(float(acct.equity), 2),
            "cash":      round(float(acct.cash), 2),
            "pnl":       round(float(acct.equity) - float(acct.last_equity), 2),
            "regime":    market_regime,
        }
    except Exception as e:
        print(f"get_account_state error: {e}")
        return {"portfolio": 0, "cash": 0, "pnl": 0, "regime": market_regime}

# ── Socket Connect (FIXED) ────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("Client connected")
    try:
        window_open = in_trading_window()
        market_open = is_market_open() if window_open else False

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

                    state["tickers"][ticker] = {
                        "price": price,
                        "buys": buys,
                        "sells": sells,
                        "signals": sigs
                    }

                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue

        socketio.emit("state", state)

    except Exception as e:
        print(f"on_connect error: {e}")

# ── Dummy placeholders (unchanged logic elsewhere) ─────────────
# NOTE: These exist only because your original paste was cut.
# Your real file already has these — keep yours.

def in_trading_window():
    now = datetime.now(NY)
    return True

def is_market_open():
    return True

def get_signals(ticker, price):
    return []