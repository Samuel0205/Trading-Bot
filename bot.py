import os, time, threading, json, sys
sys.stdout.reconfigure(line_buffering=True)

from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from ops_reviewer import _ops_stats, build_ops_findings, analyze_with_claude as _ai_analyze

from database import (
    init_db, save_trade_open, save_trade_close, save_partial_exit,
    get_recent_trades, get_trade_stats_from_db, save_signal_performance,
    get_signal_win_rates, save_portfolio_snapshot, get_portfolio_history,
    save_alert, get_alerts, acknowledge_alerts, save_price_history,
    load_price_history, save_scan_result, is_macro_blackout,
    get_open_position_stops
)
from macro import (
    seed_macro_calendar, check_earnings_risk, is_macro_blackout_today,
    get_sector_momentum, scan_unusual_volume, get_macro_status, get_hot_sectors
)

def get_scanner():
    try:
        from scanner import run_full_scan, SEED_UNIVERSE, EXTENDED_UNIVERSE
        return run_full_scan, list(dict.fromkeys(SEED_UNIVERSE + EXTENDED_UNIVERSE))
    except Exception as e:
        print(f"Scanner import error: {e}")
        return None, []

def get_predictor():
    try:
        from predictions import run_predictions, calculate_stops
        return run_predictions, calculate_stops
    except Exception as e:
        print(f"Predictor import error: {e}")
        return None, None

def get_backtester():
    try:
        from backtest import run_backtest
        return run_backtest
    except Exception as e:
        print(f"Backtester import error: {e}")
        return None

def get_catalyst_tracker():
    try:
        from catalyst_tracker import get_catalyst_score, get_catalyst_tickers, get_edgar_scores
        return get_catalyst_score, get_catalyst_tickers, get_edgar_scores
    except Exception as e:
        print(f"Catalyst tracker import error: {e}")
        return None, None, None

# ── API ───────────────────────────────────────────────────────
API_KEY    = os.environ.get("APCA_API_KEY_ID")
SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
LIVE_MODE  = os.environ.get("LIVE_TRADING", "false").lower() == "true"
BASE_URL   = "https://api.alpaca.markets" if LIVE_MODE else "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys.")

print(f"=== MODE: {'🔴 LIVE TRADING' if LIVE_MODE else '📄 PAPER TRADING'} ===")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Config ────────────────────────────────────────────────────
DEFAULT_MAX_ACCOUNT  = float(os.environ.get("MAX_ACCOUNT", "30.00"))
MAX_TRADE_PCT        = 0.45
STOP_LOSS_PCT        = 0.05
TAKE_PROFIT_PCT      = 0.12
PARTIAL_EXIT_PCT     = 0.06   # NEW: sell half position at 6% gain
INTERVAL             = 30
MIN_GRADE            = ["A","B","C","D"]
COOLDOWN_STOP        = 900
COOLDOWN_PROFIT      = 600
COOLDOWN_SIGNAL      = 300
TRADING_START_H      = 9
TRADING_START_M      = 35
TRADING_END_H        = 15
TRADING_END_M        = 30
NO_NEW_ENTRY_MINS    = 20    # NEW: no new buys in first 20 min (9:35–9:55)
EOD_TIGHTEN_MINS     = 30   # NEW: tighter thresholds last 30 min (3:00–3:30)
SCAN_HOURS           = [10, 12]
PRED_HOURS           = [9, 10, 11, 12, 13, 14]   # refresh every hour during trading day
PRED_MAX_AGE_SECS    = 5400                        # 90 min — treat stale prediction as neutral
MACRO_REFRESH_HOURS  = [8, 12]
MIN_PROFIT_PCT       = 0.04
MAX_DAILY_TRADES     = 10
PDT_MAX              = 3      # PDT rule: max 3 day trades per rolling 5-business-day window
PDT_ACCOUNT_LIMIT    = 25_000 # PDT only applies to accounts below this equity threshold
TRADE_WINDOW_SECS    = 7200   # 2-hour rolling window for trade spreading
MAX_WINDOW_TRADES    = 4      # max new entries in any 2-hour window
GAP_MIN_PCT          = 2.0
GAP_MAX_PCT          = 15.0
GAP_MIN_RVOL          = 1.5
GAP_GO_MIN_PCT        = 5.0    # min gap% to trigger gap-and-go entry
GAP_ENTRY_WINDOW_MINS = 90     # don't enter gap plays after 90 min post-open
TRAILING_STOP_GAP_PCT = 0.03   # trail stop 3% below highest price for gap plays
BREAKEVEN_TRIGGER_PCT = 0.08   # move stop to breakeven when up 8%
TRAILING_TRIGGER_PCT  = 0.15   # activate 3% trail once up 15%
RVOL_THRESHOLD        = 1.2
CHOP_BLOCK_THRESHOLD  = 2      # stop a ticker for rest of day after this many stop losses
RVOL_PROMOTE_MIN     = 2.5   # NEW: promote ticker to active list if rvol exceeds this
BASE_BUY_THRESHOLD   = 2.0
BASE_SELL_THRESHOLD  = 2.0
PRED_STRONG_BUY      = 40
PRED_SKIP            = -35
PRED_NEED_CONF       = -10
DB_SNAPSHOT_INTERVAL = 3600

# NEW: RSI veto thresholds — hard blocks regardless of other signals
RSI_OVERBOUGHT_VETO  = 75   # block buys when RSI above this
RSI_OVERSOLD_VETO    = 25   # block sells when RSI below this

# NEW: price extension veto — don't chase extended moves
PRICE_EXTENSION_VETO = 0.08  # block buys if price > 8% above 20-bar mean

BASE_SIGNAL_WEIGHTS = {
    "MA Crossover":   0.8,
    "RSI":            1.2,
    "Bollinger":      1.0,
    "VWAP":           1.5,
    "MACD":           1.0,
    "Mean Reversion": 0.8,
}

FALLBACK_TICKERS = ["SOFI","HOOD","NIO","MARA","RIOT","PLTR","AFRM","DKNG","SNAP","RIVN"]

# ── State ─────────────────────────────────────────────────────
price_history     = {}
volume_history    = {}
trade_log         = []
scan_results      = {"today":[],"yesterday":[],"scanned_at":None}
prediction_cache  = {}
backtest_cache    = {}
macro_status      = {}
active_tickers    = list(FALLBACK_TICKERS)
open_positions    = {}     # { ticker: {entry, stop, target, qty, partial_done, ...} }
market_regime     = "ranging"
cooldowns         = {}
ticker_grades     = {}
gap_candidates    = {}
catalyst_cache    = {}
rvol_cache        = {}
_daily_stop_counts     = {}   # ticker → {date_str: count}  (anti-chop gate)
_daily_ticker_dt_counts = {}  # ticker → {date_str: count}  (day trades per ticker per day)
unusual_volume    = []
daily_trade_count = 0
daily_trade_date  = None
_trade_window_times  = []     # buy timestamps in rolling window (guarded by _trade_count_lock)
last_db_snapshot  = 0
signal_weights    = dict(BASE_SIGNAL_WEIGHTS)

# Pending limit orders — keyed by Alpaca order_id
# Each entry: {ticker, qty, stop, target, atr, active_signals,
#              price, pred_score, rvol, submitted_at, ts}
_pending_orders      = {}
_pending_buy_tickers = set()    # tickers with an open limit order (blocks duplicate entries)
_pending_lock        = threading.Lock()
ORDER_FILL_TIMEOUT   = 90       # seconds before an unfilled limit order is cancelled

# PDT check cache — avoids hitting Alpaca API on every buy attempt
_pdt_cache      = {"count": 0, "ts": 0.0, "window_start": None, "reset_date": None}
_PDT_CACHE_TTL  = 300   # 5 minutes

_blackout_cache      = {"date": None, "result": (False, None)}
_blackout_cache_lock = threading.Lock()
_regime_lock         = threading.Lock()
_trade_count_lock    = threading.Lock()
_positions_lock      = threading.Lock()   # guards open_positions check-then-set

# ── Health tracking ───────────────────────────────────────────
_thread_heartbeats = {}   # name → Unix timestamp of last tick
_thread_errors     = {}   # name → last exception string
_bot_start_time    = time.time()
_api_last_success  = 0.0

NY = pytz.timezone("America/New_York")


def _reset_ops_stats_if_new_day():
    """Reset _ops_stats at the start of each new trading day."""
    try:
        today = now_et().date()
        if _ops_stats["date"] != today:
            _ops_stats.update({
                "date":                today,
                "daily_limit_hits":    {},
                "filter_blocks":       {},
                "finbert_errors":      0,
                "finbert_method":      None,
                "scan_tickers":        [],
                "blocked_buy_tickers": set(),
            })
    except Exception:
        pass

# ── Time helpers ──────────────────────────────────────────────

def now_et():
    return datetime.now(NY)

def _hb(name):
    _thread_heartbeats[name] = time.time()

def in_trading_window():
    now  = now_et()
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return (TRADING_START_H*60+TRADING_START_M) <= mins < (TRADING_END_H*60+TRADING_END_M)

def is_market_open():
    try:
        return api.get_clock().is_open
    except Exception as e:
        print(f"is_market_open error: {e}")
        return False

def mins_since_open():
    """Minutes elapsed since 9:30 AM ET today."""
    now = now_et()
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((now - open_time).total_seconds() / 60))

def mins_until_close():
    """Minutes until 3:30 PM ET today."""
    now = now_et()
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return max(0, int((close_time - now).total_seconds() / 60))

# NEW: time-of-day entry gate
def can_enter_new_position():
    """
    Block new buys:
    - First 20 minutes after open (9:30–9:50): too volatile, fakeouts common
    - Last 5 minutes (3:25–3:30): EOD volatility, hard to fill cleanly
    """
    elapsed = mins_since_open()
    remaining = mins_until_close()
    if elapsed < NO_NEW_ENTRY_MINS:
        print(f"  Time gate: {NO_NEW_ENTRY_MINS - elapsed}min until entry allowed")
        return False
    if remaining < 5:
        print(f"  Time gate: too close to close ({remaining}min left)")
        return False
    return True

def get_threshold_multiplier():
    """
    NEW: tighten vote thresholds in last 30 minutes.
    Requires stronger consensus before entering a position late in the day.
    """
    remaining = mins_until_close()
    if remaining <= EOD_TIGHTEN_MINS:
        return 1.4   # 40% harder to get a signal late day
    return 1.0

# ── Blackout cache ─────────────────────────────────────────────

def cached_blackout_today():
    today = now_et().strftime("%Y-%m-%d")
    with _blackout_cache_lock:
        if _blackout_cache["date"] == today:
            return _blackout_cache["result"]
        result = is_macro_blackout_today()
        _blackout_cache["date"]   = today
        _blackout_cache["result"] = result
        return result

# ── Signal weight self-learning ───────────────────────────────

def update_signal_weights_from_db():
    global signal_weights
    try:
        perf = get_signal_win_rates()
        new_weights = {}
        for name, base in BASE_SIGNAL_WEIGHTS.items():
            if name not in perf or perf[name]["total"] < 8:
                new_weights[name] = base
                continue
            wr = perf[name]["win_rate"] / 100
            if   wr > 0.65: adj = min(base * 1.30, base + 0.6)
            elif wr > 0.55: adj = min(base * 1.10, base + 0.2)
            elif wr < 0.35: adj = max(base * 0.70, base - 0.4)
            elif wr < 0.45: adj = max(base * 0.90, base - 0.2)
            else:           adj = base
            new_weights[name] = round(adj, 3)
        if new_weights != signal_weights:
            print(f"Signal weights updated: {new_weights}")
            signal_weights = new_weights
    except Exception as e:
        print(f"update_signal_weights error: {e}")

# ── Account ───────────────────────────────────────────────────

def get_account():
    global _api_last_success
    try:
        acct = api.get_account()
        _api_last_success = time.time()
        return acct
    except Exception as e:
        print(f"get_account error: {e}")
        return None

def get_account_size():
    acct = get_account()
    if acct:
        return float(acct.equity)
    return DEFAULT_MAX_ACCOUNT

def get_available_cash():
    acct = get_account()
    if not acct: return 0
    if LIVE_MODE:
        return float(acct.non_marginable_buying_power)
    return min(float(acct.cash), float(acct.buying_power))

def get_account_state():
    try:
        acct = get_account()
        if not acct:
            return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime,"live":LIVE_MODE,
                    "open_positions":{}}

        # NEW: include unrealized P&L per open position
        positions_detail = {}
        try:
            for pos in api.list_positions():
                sym = pos.symbol
                positions_detail[sym] = {
                    "qty":         int(pos.qty),
                    "entry":       round(float(pos.avg_entry_price), 2),
                    "current":     round(float(pos.current_price), 2),
                    "unrealized":  round(float(pos.unrealized_pl), 2),
                    "unrealized_pct": round(float(pos.unrealized_plpc) * 100, 2),
                    "market_value": round(float(pos.market_value), 2),
                }
        except: pass

        return {
            "portfolio":      round(float(acct.equity), 2),
            "cash":           round(float(acct.cash), 2),
            "pnl":            round(float(acct.equity) - float(acct.last_equity), 2),
            "regime":         market_regime,
            "live":           LIVE_MODE,
            "open_positions": positions_detail,
        }
    except Exception as e:
        print(f"get_account_state error: {e}")
        return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime,"live":LIVE_MODE,
                "open_positions":{}}

# ── Scaling ───────────────────────────────────────────────────

def get_price_ceiling(account_size=None):
    if account_size is None: account_size = get_account_size()
    if account_size > 50_000: return min(account_size * 0.45, 50.00)
    if account_size > 10_000: return min(account_size * 0.45, 25.00)
    return max(min(account_size * 0.45, 10.00), 0.60)

def get_price_floor(account_size=None):
    if account_size is None: account_size = get_account_size()
    if account_size > 10_000: return 3.00
    if account_size > 100:    return 2.00
    if account_size > 20:     return 1.00
    return 0.50

def get_min_volume(account_size=None):
    if account_size is None: account_size = get_account_size()
    # Calibrated for IEX exchange feed (2-3% of consolidated NASDAQ/NYSE volume).
    # IEX shows ~10k shares/day for stocks that trade 500k-1M real shares/day.
    # The prediction score and RVOL gates handle real liquidity filtering.
    if account_size > 5000: return 2_000
    if account_size > 1000: return 1_000
    if account_size > 200:  return 500
    return 200

# ── PDT compliance ────────────────────────────────────────────

def _pdt_window_start():
    """First date of the rolling 5-business-day PDT window (today inclusive)."""
    dt  = now_et().date()
    bds = 0
    while bds < 4:          # back 4 more BDs so window = today + 4 prior BDs = 5 total
        dt -= timedelta(days=1)
        if dt.weekday() < 5:
            bds += 1
    return dt

def _add_business_days(dt, n):
    """Return the date exactly n business days after dt."""
    result = dt
    count  = 0
    while count < n:
        result += timedelta(days=1)
        if result.weekday() < 5:
            count += 1
    return result

def count_rolling_day_trades():
    """
    Count same-security same-day round-trips (day trades) in the rolling
    5-business-day PDT window.  Alpaca flags accounts that exceed 3.
    Works in both LIVE and PAPER mode (paper fills appear in the orders API).

    Returns (count, window_start_str, reset_date_str).
    reset_date_str is a human-readable date when the oldest trade rolls off.
    Cached for _PDT_CACHE_TTL seconds.
    """
    if time.time() - _pdt_cache["ts"] < _PDT_CACHE_TTL:
        return (_pdt_cache["count"],
                _pdt_cache["window_start"],
                _pdt_cache["reset_date"])
    try:
        win_start = _pdt_window_start()
        after_dt  = NY.localize(datetime.combine(win_start, datetime.min.time()))
        orders    = api.list_orders(
            status="filled",
            after=after_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=100, direction="asc"
        )
        buys  = set()
        sells = set()
        for o in orders:
            fa = o.filled_at
            d  = fa.strftime("%Y-%m-%d") if hasattr(fa, "strftime") else (fa[:10] if fa else "")
            k = (o.symbol, d)
            if o.side == "buy":  buys.add(k)
            if o.side == "sell": sells.add(k)
        round_trips = sorted(buys & sells, key=lambda x: x[1])
        count = len(round_trips)
        reset_date_str = None
        if round_trips:
            oldest_dt      = datetime.strptime(round_trips[0][1], "%Y-%m-%d").date()
            reset_dt       = _add_business_days(oldest_dt, 5)
            reset_date_str = reset_dt.strftime("%a %b %-d")
        _pdt_cache["count"]        = count
        _pdt_cache["ts"]           = time.time()
        _pdt_cache["window_start"] = win_start.strftime("%Y-%m-%d")
        _pdt_cache["reset_date"]   = reset_date_str
        return count, win_start.strftime("%Y-%m-%d"), reset_date_str
    except Exception as e:
        print(f"PDT count error: {e}")
        return (_pdt_cache["count"],
                _pdt_cache.get("window_start"),
                _pdt_cache.get("reset_date"))

def is_day_trade_safe():
    count, _, _ = count_rolling_day_trades()
    return count < MAX_DAILY_TRADES

# ── Correlation helper ────────────────────────────────────────

def _pearson(a, b, n=30):
    """Pearson correlation over the last n bars. Returns 0 if insufficient data."""
    n = min(len(a), len(b), n)
    if n < 10:
        return 0.0
    a, b   = a[-n:], b[-n:]
    ma, mb = sum(a)/n, sum(b)/n
    cov    = sum((x-ma)*(y-mb) for x,y in zip(a,b)) / n
    sa     = (sum((x-ma)**2 for x in a) / n) ** 0.5
    sb     = (sum((y-mb)**2 for y in b) / n) ** 0.5
    return cov / (sa*sb) if sa > 0 and sb > 0 else 0.0

def is_correlated_with_open_position(ticker, threshold=0.75):
    """
    Returns True if the candidate ticker's recent price moves are highly
    correlated with any currently open position.  Prevents doubling up on
    the same underlying bet (e.g. MARA + RIOT both tracking Bitcoin).
    """
    candidate = price_history.get(ticker, [])
    if len(candidate) < 10:
        return False
    for pos_ticker in list(open_positions.keys()):
        if pos_ticker == ticker:
            continue
        pos_prices = price_history.get(pos_ticker, [])
        if len(pos_prices) < 10:
            continue
        corr = _pearson(candidate, pos_prices)
        if corr > threshold:
            print(f"  {ticker} corr={corr:.2f} with open {pos_ticker} — skipping (concentration risk)")
            return True
    return False

# ── Cooldowns ─────────────────────────────────────────────────

def set_cooldown(ticker, reason="signal"):
    d = {"stop_loss":COOLDOWN_STOP,"take_profit":COOLDOWN_PROFIT,
         "signal":COOLDOWN_SIGNAL,"eod_close":COOLDOWN_SIGNAL,
         "partial":COOLDOWN_SIGNAL}
    cooldowns[ticker] = {"until": time.time()+d.get(reason, COOLDOWN_SIGNAL), "reason": reason}

def is_on_cooldown(ticker):
    cd = cooldowns.get(ticker)
    return bool(cd and time.time() < cd["until"])

def cooldown_remaining(ticker):
    cd = cooldowns.get(ticker)
    if not cd: return 0
    return max(0, int(cd["until"] - time.time()))

# ── Indicators ────────────────────────────────────────────────

def calc_rsi(prices, period=14):
    if len(prices) < period+1: return 50
    gains = losses = 0
    for i in range(-period, 0):
        d = prices[i] - prices[i-1]
        if d > 0: gains += d
        else:     losses += abs(d)
    if losses == 0: return 100
    return 100-(100/(1+gains/losses))

def calc_ma(prices, n):
    s = prices[-n:] if len(prices) >= n else prices
    return sum(s)/len(s)

def calc_bollinger(prices, n=20):
    s    = prices[-n:] if len(prices) >= n else prices
    mean = sum(s)/len(s)
    std  = (sum((v-mean)**2 for v in s)/len(s))**0.5
    return mean, mean+2*std, mean-2*std

def calc_vwap(prices, volumes):
    if not prices or not volumes or sum(volumes)==0:
        return prices[-1] if prices else 0
    return sum(p*v for p,v in zip(prices,volumes))/sum(volumes)

def calc_macd(prices):
    def ema(data, period):
        if len(data)<period: return data[-1] if data else 0
        k=2/(period+1); val=sum(data[:period])/period
        for p in data[period:]: val=p*k+val*(1-k)
        return val
    if len(prices)<26: return 0
    return ema(prices,12)-ema(prices,26)

def get_signals(ticker, price):
    hist = price_history.get(ticker, [])
    vols = volume_history.get(ticker, [])
    n    = len(hist)
    if n < 5:
        return [{
            "name":k,"action":"hold","signal":50,
            "veto":False,"veto_reason":""
        } for k in signal_weights]

    rsi       = calc_rsi(hist)
    ma50      = calc_ma(hist, min(50, n))
    ma200     = calc_ma(hist, min(200, n))
    mean, upper, lower = calc_bollinger(hist)
    vwap      = calc_vwap(hist[-78:], vols[-78:])
    macd_line = calc_macd(hist)
    z         = (price-mean)/max((upper-mean), 0.01)
    # Regime no longer hard-gates buys/sells — threshold multipliers in make_decision
    # handle it instead. Strong individual stocks can always be bought or sold.
    ok_buy  = True
    ok_sell = True
    ma_conf   = min(1.0, n/20)

    # NEW: veto conditions — hard blocks independent of vote weights
    rsi_overbought    = rsi > RSI_OVERBOUGHT_VETO
    rsi_oversold      = rsi < RSI_OVERSOLD_VETO
    price_extended    = (price - mean) / max(mean, 0.01) > PRICE_EXTENSION_VETO

    def act(bc, sc, veto_buy=False, veto_sell=False):
        if bc and ok_buy  and not veto_buy:  return "buy"
        if sc and ok_sell and not veto_sell: return "sell"
        return "hold"

    signals = [
        {
            "name":"MA Crossover",
            "action": act(ma50>ma200*1.005, ma200>ma50*1.005,
                          veto_buy=rsi_overbought or price_extended) if ma_conf>0.5 else "hold",
            "signal": min(95, 50+(ma50-ma200)/max(ma200,1)*600),
            "veto":   (rsi_overbought or price_extended) and ma50>ma200*1.005,
            "veto_reason": "RSI overbought" if rsi_overbought else ("extended" if price_extended else ""),
        },
        {
            "name":"RSI",
            "action": act(rsi<35, rsi>65),
            "signal": 100-rsi,
            "veto":   False, "veto_reason": "",
        },
        {
            "name":"Bollinger",
            "action": act(price<lower*0.99, price>upper*1.01,
                          veto_buy=rsi_overbought or price_extended,
                          veto_sell=rsi_oversold),
            "signal": max(5,min(95,50-z*40)),
            "veto":   (rsi_overbought or price_extended) and price<lower*0.99,
            "veto_reason": "RSI overbought" if rsi_overbought else ("extended" if price_extended else ""),
        },
        {
            "name":"VWAP",
            "action": act(price>vwap*1.001, price<vwap*0.999,
                          veto_buy=rsi_overbought or price_extended),
            "signal": min(95,max(5,50+(price-vwap)/max(vwap,1)*500)),
            "veto":   (rsi_overbought or price_extended) and price>vwap*1.001,
            "veto_reason": "RSI overbought" if rsi_overbought else ("extended" if price_extended else ""),
        },
        {
            "name":"MACD",
            "action": act(macd_line>0, macd_line<0,
                          veto_buy=rsi_overbought or price_extended),
            "signal": min(95,max(5,50+macd_line*10)),
            "veto":   (rsi_overbought or price_extended) and macd_line>0,
            "veto_reason": "RSI overbought" if rsi_overbought else ("extended" if price_extended else ""),
        },
        {
            "name":"Mean Reversion",
            "action": act(price<mean*0.96, price>mean*1.04,
                          veto_buy=rsi_overbought,
                          veto_sell=rsi_oversold),
            "signal": min(92,max(8,50+(mean-price)/max(mean,1)*250)),
            "veto":   rsi_overbought and price<mean*0.96,
            "veto_reason": "RSI overbought" if rsi_overbought else "",
        },
    ]

    # Global buy veto: if RSI is overbought AND price is extended, force all buys to hold
    if rsi_overbought and price_extended:
        for s in signals:
            if s["action"] == "buy":
                s["action"] = "hold"
                s["veto"]   = True
                s["veto_reason"] = f"RSI={rsi:.0f} + price {((price-mean)/mean*100):.1f}% above mean"

    return signals

def weighted_vote(signals):
    buy_w = sell_w = 0.0
    for s in signals:
        w = signal_weights.get(s.get("name",""), 1.0)
        if   s.get("action") == "buy":  buy_w  += w
        elif s.get("action") == "sell": sell_w += w
    return round(buy_w, 2), round(sell_w, 2)

# ── Regime ────────────────────────────────────────────────────

def update_market_regime():
    global market_regime
    # Use broad market ETFs — NOT individual meme stocks which are almost always
    # in a downtrend regardless of the broader market, causing ok_buy=False every day.
    RTICKERS = ["SPY", "QQQ"]
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=15)
        ss    = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        es    = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        closes = None
        regime_src = None
        for ticker in RTICKERS:
            try:
                bars = api.get_bars(ticker,"1Day",start=ss,end=es,limit=12,feed="iex").df
                if bars is None or bars.empty: continue
                if hasattr(bars.index,'levels'):
                    if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                    else: continue
                if len(bars)>=5:
                    closes=list(bars["close"]); regime_src=ticker; break
            except: continue
        if closes is None:
            all_p=[p for t in active_tickers for p in price_history.get(t,[])[-10:]]
            closes=all_p if len(all_p)>=5 else None
        if closes is None:
            print(f"Regime: no data, keeping {market_regime}"); return
        ma5    = calc_ma(closes, min(5,  len(closes)))
        ma10   = calc_ma(closes, min(10, len(closes)))
        latest = closes[-1]
        with _regime_lock:
            if   ma5>ma10*1.005 and latest>ma5: market_regime="trending_up"
            elif ma5<ma10*0.995 and latest<ma5: market_regime="trending_down"
            else:                               market_regime="ranging"
        print(f"Regime: {market_regime} (source: {regime_src}, ma5={ma5:.2f} ma10={ma10:.2f})")
    except Exception as e:
        print(f"Regime error: {e}")

# ── Filters ───────────────────────────────────────────────────

# Cache: volume check per ticker per day to avoid fetching bars on every buy attempt
_vol_check_cache = {}  # { ticker: (date, passed_bool) }

# Cache: earnings risk per ticker per day — avoids news API call every 30s
_earnings_cache = {}   # { ticker: (date, (risk_level, adj, detail)) }

def passes_filters(ticker, price, account_size=None):
    if account_size is None: account_size = get_account_size()
    floor   = get_price_floor(account_size)
    ceiling = get_price_ceiling(account_size)
    min_vol = get_min_volume(account_size)
    if not(floor<=price<=ceiling):
        print(f"  {ticker} ${price:.2f} outside range ${floor}–${ceiling}")
        try:
            _ops_stats["filter_blocks"]["price_range"] = _ops_stats["filter_blocks"].get("price_range", 0) + 1
        except Exception:
            pass
        return False
    grade = ticker_grades.get(ticker)
    if grade and grade not in MIN_GRADE:
        print(f"  {ticker} grade {grade} excluded"); return False

    # rvol check — only gate if we have a meaningful reading (>= 0.10x).
    # IEX free tier only covers ~2-3% of NASDAQ/NYSE volume, so liquid stocks
    # like XPEV or CLF often show 0.01-0.04x — that's IEX noise, not real low volume.
    # Below 0.10x → treat as no data (same as None → allow through).
    # 0.10x–RVOL_THRESHOLD → genuinely thin stock → block.
    rvol = rvol_cache.get(ticker)
    if rvol is not None and rvol >= 0.10 and rvol < RVOL_THRESHOLD:
        print(f"  {ticker} rvol {rvol:.2f}x low (< {RVOL_THRESHOLD}x)")
        try:
            _ops_stats["filter_blocks"]["rvol"] = _ops_stats["filter_blocks"].get("rvol", 0) + 1
        except Exception:
            pass
        return False

    # Gap plays with strong intraday RVOL bypass the historical daily volume check
    if (gap_candidates.get(ticker, {}).get("direction") == "up" and
            (rvol_cache.get(ticker) or 0) >= 2.0):
        return True

    # Volume check — cached per day to avoid hitting Alpaca on every 30s tick
    today_str = now_et().strftime("%Y-%m-%d")
    cached_vol = _vol_check_cache.get(ticker)
    if cached_vol and cached_vol[0] == today_str:
        if not cached_vol[1]:
            print(f"  {ticker} low volume (cached)")
            try:
                _ops_stats["filter_blocks"]["volume"] = _ops_stats["filter_blocks"].get("volume", 0) + 1
            except Exception:
                pass
            return False
        return True
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=5)
        bars  = api.get_bars(ticker,"1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=5, feed="iex").df
        if bars is None or bars.empty:
            print(f"  {ticker} volume check: IEX returned no bars — low volume assumed")
            _vol_check_cache[ticker] = (today_str, False)
            try:
                _ops_stats["filter_blocks"]["volume"] = _ops_stats["filter_blocks"].get("volume", 0) + 1
            except Exception:
                pass
            return False
        if hasattr(bars.index,'levels'):
            if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
            else:
                _vol_check_cache[ticker] = (today_str, False)
                try:
                    _ops_stats["filter_blocks"]["volume"] = _ops_stats["filter_blocks"].get("volume", 0) + 1
                except Exception:
                    pass
                return False
        passed = float(bars["volume"].mean()) >= min_vol
        _vol_check_cache[ticker] = (today_str, passed)
        if not passed:
            print(f"  {ticker} low volume")
            try:
                _ops_stats["filter_blocks"]["volume"] = _ops_stats["filter_blocks"].get("volume", 0) + 1
            except Exception:
                pass
            return False
        return True
    except Exception as e:
        print(f"  Filter error {ticker}: {e}"); return False

# ── Position sizing ───────────────────────────────────────────

def position_size(price, account_size=None, pred_score=0, rvol=1.0, is_gap_play=False):
    try:
        if account_size is None: account_size = get_account_size()
        usable = get_available_cash()
        if usable <= 0: return 0
        if is_gap_play:
            pct = 0.65   # larger allocation for catalyst-driven gap plays
        elif pred_score >= PRED_STRONG_BUY: pct = 0.60
        elif pred_score >= 0:               pct = MAX_TRADE_PCT
        else:                               pct = 0.30
        if not is_gap_play:
            if   rvol >= 3.0: pct = min(pct*1.15, 0.65)
            elif rvol >= 2.0: pct = min(pct*1.05, 0.55)
        if   price < 1.00: pct *= 0.80
        elif price < 2.00: pct *= 0.90
        # Hard cap on single-position concentration as account grows.
        # Small accounts need concentration to afford stocks; large accounts
        # don't — a 45% position in a $100k account is $45k in one volatile stock.
        if is_gap_play:
            if   account_size > 10_000: pct = min(pct, 0.15)
            elif account_size > 1_000:  pct = min(pct, 0.30)
        else:
            if   account_size > 10_000: pct = min(pct, 0.10)
            elif account_size > 1_000:  pct = min(pct, 0.20)
            elif account_size > 100:    pct = min(pct, 0.35)
        max_spend = usable * pct
        if max_spend < price:
            print(f"  Can't afford ${price:.2f} (max ${max_spend:.2f})"); return 0
        return max(0, int(max_spend/price))
    except Exception as e:
        print(f"  position_size error: {e}"); return 0

# ── Stops + partial exits ─────────────────────────────────────

def check_stops(ticker, price):
    pos = open_positions.get(ticker)
    if not pos: return

    entry         = pos["entry"]
    gain_pct      = (price - entry) / entry

    # Track highest price for gap play trailing stops
    if pos.get("is_gap_play") and price > pos.get("highest_price", entry):
        open_positions[ticker]["highest_price"] = price

    # partial exit — sell half at PARTIAL_EXIT_PCT gain, let rest run
    if not pos.get("partial_done") and gain_pct >= PARTIAL_EXIT_PCT:
        half_qty = max(1, pos["qty"] // 2)
        try:
            api.submit_order(symbol=ticker, qty=half_qty, side="sell",
                             type="market", time_in_force="day")
            partial_pnl = round((price - entry) * half_qty, 2)
            open_positions[ticker]["qty"]          = pos["qty"] - half_qty
            open_positions[ticker]["partial_done"] = True
            # Move stop to breakeven after partial exit
            open_positions[ticker]["stop"]         = round(entry * 1.005, 3)
            print(f"  PARTIAL EXIT {half_qty}x {ticker} @ ${price:.2f} "
                  f"pnl=${partial_pnl:+.2f} | stop → breakeven")
            ts_partial = int(time.time()*1000)
            trade_log.insert(0, {
                "type":"SELL","ticker":ticker,"qty":half_qty,
                "price":round(price,2),"pnl":partial_pnl,
                "reason":"partial_exit","ts":ts_partial
            })
            try:
                save_partial_exit(ticker, half_qty, entry, price, partial_pnl, ts_partial)
            except Exception as e:
                print(f"  DB save_partial_exit error: {e}")
            save_alert("INFO",
                f"PARTIAL EXIT {half_qty}x {ticker} @ ${price:.2f} +${partial_pnl:.2f}",
                ticker)
        except Exception as e:
            print(f"  Partial exit error {ticker}: {e}")
        return

    if pos.get("is_gap_play"):
        highest = pos.get("highest_price", price)
        if gain_pct >= TRAILING_TRIGGER_PCT:
            trail_stop = round(highest * (1 - TRAILING_STOP_GAP_PCT), 3)
            if trail_stop > pos["stop"]:
                open_positions[ticker]["stop"] = trail_stop
                print(f"  GAP trail {ticker} → ${trail_stop:.3f} (high=${highest:.2f})")
        elif gain_pct >= BREAKEVEN_TRIGGER_PCT:
            be_stop = round(entry * 1.005, 3)
            if be_stop > pos["stop"]:
                open_positions[ticker]["stop"] = be_stop
                print(f"  GAP breakeven {ticker} → ${be_stop:.3f}")
    else:
        # Standard trailing stop: once up 5%, move stop to entry+1%
        if gain_pct > 0.05:
            new_stop = max(pos["stop"], round(entry * 1.01, 3))
            if new_stop > pos["stop"]:
                open_positions[ticker]["stop"] = new_stop
                print(f"  Trail stop {ticker} → ${new_stop:.3f}")

    if   price <= pos["stop"]:   force_sell(ticker, price, reason="stop_loss")
    elif price >= pos["target"]: force_sell(ticker, price, reason="take_profit")

def force_sell(ticker, price, reason="stop_loss"):
    try:
        try:
            pos_api = api.get_position(ticker)
        except Exception as pos_err:
            # Position doesn't exist in Alpaca (closed externally, 404, etc.)
            # Remove from tracking so the bot stops trying to manage it.
            with _positions_lock:
                open_positions.pop(ticker, None)
            print(f"  {ticker}: position not found in Alpaca ({pos_err}), removed from tracking")
            return
        qty     = int(pos_api.qty)
        if qty <= 0:
            with _positions_lock:
                open_positions.pop(ticker, None)
            return
        entry   = float(pos_api.avg_entry_price)
        pnl     = round((price - entry) * qty, 2)
        api.submit_order(symbol=ticker, qty=qty, side="sell",
                         type="market", time_in_force="day")
        was_win = pnl > 0
        with _positions_lock:
            pos_data = open_positions.pop(ticker, {})
        active_sigs = pos_data.get("active_signals", [])

        try:
            save_trade_close(ticker, price, pnl, reason, int(time.time()*1000))
            for sig in active_sigs:
                save_signal_performance(sig, was_win, ticker, pnl)
        except Exception as e:
            print(f"  DB save error: {e}")

        update_signal_weights_from_db()

        trade_log.insert(0, {
            "type":"SELL","ticker":ticker,"qty":qty,
            "price":round(price,2),"pnl":pnl,"reason":reason,
            "ts":int(time.time()*1000)
        })
        set_cooldown(ticker, reason)

        today_str = now_et().strftime("%Y-%m-%d")

        # Anti-chop gate: count stop losses per ticker per day
        if reason == "stop_loss":
            _daily_stop_counts.setdefault(ticker, {})
            _daily_stop_counts[ticker][today_str] = (
                _daily_stop_counts[ticker].get(today_str, 0) + 1
            )
            stops_today = _daily_stop_counts[ticker][today_str]
            if stops_today >= CHOP_BLOCK_THRESHOLD:
                print(f"  CHOP BLOCK: {ticker} stopped out {stops_today}x today — "
                      f"blocked for the rest of the day")

        # Track per-ticker day trades (opened and closed same calendar day)
        if reason != "eod_close":
            if pos_data.get("opened_date") == today_str:
                _daily_ticker_dt_counts.setdefault(ticker, {})
                _daily_ticker_dt_counts[ticker][today_str] = (
                    _daily_ticker_dt_counts[ticker].get(today_str, 0) + 1
                )

        wr   = get_trade_stats_from_db().get("win_rate", 0)
        mode = "🔴 LIVE" if LIVE_MODE else "PAPER"
        print(f"{mode} SELL {qty}x {ticker} @ ${price:.2f} "
              f"| {reason} | PnL ${pnl:+.2f} | WR:{wr:.0f}%")

        if abs(pnl) > 0.01:
            save_alert("INFO" if was_win else "WARN",
                       f"{'WIN' if was_win else 'LOSS'} ${pnl:+.2f} on {ticker} ({reason})",
                       ticker)
    except Exception as e:
        print(f"Force sell error {ticker}: {e}")

def close_all_positions_eod():
    # Cancel pending limit orders first so they don't fill after close
    try:
        api.cancel_all_orders()
        with _pending_lock:
            _pending_orders.clear()
            _pending_buy_tickers.clear()
        print("EOD: pending orders cancelled")
    except Exception as e:
        print(f"EOD cancel orders error: {e}")
    try:
        for pos in api.list_positions():
            force_sell(pos.symbol, float(pos.current_price), reason="eod_close")
        print("EOD: all positions closed")
    except Exception as e:
        print(f"EOD close error: {e}")

# ── Decision engine ───────────────────────────────────────────

def make_decision(ticker, signals, price):
    buy_w, sell_w = weighted_vote(signals)
    pred    = prediction_cache.get(ticker, {})
    pscore  = pred.get("score", 0)
    tf_bias = pred.get("tf_bias", 0)
    rvol    = rvol_cache.get(ticker, 1.0)
    is_gap  = ticker in gap_candidates

    # Staleness gate: if prediction is >90 min old, treat as neutral before allowing buys.
    # This prevents buying against an hours-old bullish score when price has reversed.
    pred_age = time.time() - pred.get("fetched_at", 0)
    if pred_age > PRED_MAX_AGE_SECS and pscore > 0:
        print(f"  {ticker}: prediction stale ({pred_age/60:.0f}min old) — capping score to 0")
        pscore = 0

    if pscore <= PRED_SKIP:
        return "hold", f"pred_skip({pscore})", buy_w, sell_w

    # EOD threshold multiplier — harder to enter near close
    tod_mult = get_threshold_multiplier()

    if   pscore >= PRED_STRONG_BUY: bt = 1.2 * tod_mult; st = 2.5 * tod_mult
    elif pscore >= 20:              bt = 1.5 * tod_mult; st = 2.5 * tod_mult
    elif pscore < PRED_NEED_CONF:   bt = 2.5 * tod_mult; st = 2.0 * tod_mult
    else:                           bt = BASE_BUY_THRESHOLD * tod_mult; st = BASE_SELL_THRESHOLD * tod_mult

    # Confidence multiplier — prediction confidence now affects entry threshold.
    # High confidence → lower bar (0.85×), low confidence → raise bar (1.20×).
    pred_conf = pred.get("confidence", "low")
    conf_mult = {"high": 0.85, "medium": 1.0, "low": 1.20}.get(pred_conf, 1.0)
    bt *= conf_mult
    st *= conf_mult

    gap_data = gap_candidates.get(ticker, {})
    if is_gap and gap_data.get("direction") == "up":
        gap_pct_val = abs(gap_data.get("gap_pct", 0))
        if (gap_pct_val >= GAP_GO_MIN_PCT and
                mins_since_open() <= GAP_ENTRY_WINDOW_MINS):
            cat     = catalyst_cache.get(ticker, {})
            vwap_ok = any(s["name"] == "VWAP" and s["action"] == "buy" for s in signals)
            if cat.get("score", 0) >= 5 and vwap_ok:
                bt = max(1.2, bt - 0.8)
                print(f"  GAP-AND-GO {ticker}: gap={gap_pct_val:.1f}% "
                      f"cat={cat.get('score',0)} vwap_ok")
            else:
                bt = max(1.0, bt - 0.3)
        else:
            bt = max(1.0, bt - 0.3)
    if rvol >= 2.0:
        bt = max(1.0, bt-0.2)  # high rvol lowers entry bar only — not exit bar
    if tf_bias == -1:
        bt += 0.8

    hot_sectors = macro_status.get("hot_sectors", [])
    if hot_sectors:
        bt = max(1.0, bt-0.1)

    # Count active vetos — if any signals are vetoed, note it
    veto_count = sum(1 for s in signals if s.get("veto"))
    if veto_count >= 2:
        # 2+ signals vetoed means the setup is compromised — require stronger vote
        bt = min(bt * 1.3, 5.0)

    # Regime-based threshold scaling — never blocks trading outright, just adjusts conviction.
    # Trending down: buy is 1.5× harder + must have strong individual setup (pred >= PRED_STRONG_BUY).
    # Trending up:   buy is 0.85× easier; sell is 1.3× harder (let winners run).
    # Ranging:       neutral.
    bt *= {"trending_up": 0.85, "ranging": 1.0, "trending_down": 1.5}.get(market_regime, 1.0)
    st *= {"trending_up": 1.3,  "ranging": 1.0, "trending_down": 0.8}.get(market_regime, 1.0)
    bt  = min(bt, 5.0); st = min(st, 5.0)

    if buy_w >= bt and not (market_regime == "trending_down" and pscore < PRED_STRONG_BUY):
        action = "buy";  reason = f"bw={buy_w:.1f}>={bt:.1f}"
    elif sell_w >= st:
        action = "sell"; reason = f"sw={sell_w:.1f}>={st:.1f}"
    elif buy_w >= bt and market_regime == "trending_down":
        action = "hold"; reason = f"downtrend:pred({pscore:+.0f})<{PRED_STRONG_BUY}"
    else:
        action = "hold"; reason = f"hold(b={buy_w:.1f},s={sell_w:.1f})"
    return action, reason, buy_w, sell_w

# ── Trade execution ───────────────────────────────────────────

def _register_filled_order(order_id, fill_price):
    """
    Called when a pending limit order has been confirmed filled.
    Creates the open_positions entry, saves to DB, and logs the trade.
    """
    with _pending_lock:
        pdata = _pending_orders.pop(order_id, None)
        if pdata:
            _pending_buy_tickers.discard(pdata["ticker"])
    if not pdata:
        return

    ticker      = pdata["ticker"]
    qty         = pdata["qty"]
    stop        = pdata["stop"]
    target      = pdata["target"]
    atr         = pdata["atr"]
    active_sigs = pdata["active_signals"]
    pred_score  = pdata["pred_score"]
    rvol        = pdata["rvol"]
    is_gap_play = pdata.get("is_gap_play", False)
    ts          = int(time.time() * 1000)

    with _positions_lock:
        if ticker in open_positions:
            return  # position was reconciled or entered another way
        open_positions[ticker] = {
            "entry": fill_price, "stop": stop, "target": target,
            "qty": qty, "atr": atr, "active_signals": active_sigs,
            "partial_done": False, "is_gap_play": is_gap_play,
            "highest_price": fill_price,
            "opened_date": now_et().strftime("%Y-%m-%d"),
            "hold_since": time.time(),
        }

    try:
        save_trade_open(ticker, qty, fill_price, stop, target, pred_score,
                        rvol, ticker in gap_candidates, active_sigs, ts)
    except Exception as e:
        print(f"  DB save_trade_open error: {e}")

    rr = round((target-fill_price)/(fill_price-stop), 2) if fill_price > stop else "?"
    trade_log.insert(0, {
        "type":"BUY","ticker":ticker,"qty":qty,
        "price":round(fill_price,2),"pnl":None,
        "stop":stop,"target":target,
        "pred_score":pred_score,"rvol":rvol,
        "gap":ticker in gap_candidates,"reason":"limit_filled",
        "ts":ts
    })
    mode = "🔴 LIVE" if LIVE_MODE else "PAPER"
    print(f"{mode} BUY FILLED {qty}x {ticker} @ ${fill_price:.2f} "
          f"SL${stop} TP${target} R:R={rr} pred={pred_score:+.0f}")
    save_alert("INFO", f"BUY {qty}x {ticker} @ ${fill_price:.2f} | pred={pred_score:+.0f}", ticker)


def check_pending_orders():
    """
    Poll Alpaca for the status of each pending limit order.
    Called from bot_loop on every tick.
    - Filled → register position, save to DB
    - Canceled/rejected/expired → roll back daily_trade_count
    - Age > ORDER_FILL_TIMEOUT → cancel it (will clean up next tick)
    """
    global daily_trade_count
    for order_id in list(_pending_orders.keys()):
        pdata = _pending_orders.get(order_id)
        if not pdata:
            continue
        try:
            order = api.get_order(order_id)
            status = order.status

            if status == "filled":
                fill_price = float(order.filled_avg_price or order.limit_price)
                _register_filled_order(order_id, fill_price)

            elif status in ("canceled", "expired", "rejected"):
                with _pending_lock:
                    removed = _pending_orders.pop(order_id, None)
                    if removed:
                        _pending_buy_tickers.discard(removed["ticker"])
                if removed:
                    with _trade_count_lock:
                        daily_trade_count = max(0, daily_trade_count - 1)
                        wts = removed.get("window_ts")
                        if wts is not None:
                            try: _trade_window_times.remove(wts)
                            except ValueError: pass
                        elif _trade_window_times:
                            _trade_window_times.pop()
                    print(f"  Limit order {status}: {removed['ticker']} — count rolled back")

            elif time.time() - pdata["submitted_at"] > ORDER_FILL_TIMEOUT:
                print(f"  Limit order timeout ({ORDER_FILL_TIMEOUT}s): cancelling {pdata['ticker']}")
                try:
                    api.cancel_order(order_id)
                except Exception:
                    pass  # status update will clean up next tick
        except Exception as e:
            print(f"  check_pending_orders error {order_id}: {e}")


def execute(ticker, action, price, signals, reason="signal"):
    global daily_trade_count, daily_trade_date
    try:
        if action == "buy" and ticker not in open_positions:
            # Block if there's already a pending limit order for this ticker
            if ticker in _pending_buy_tickers:
                print(f"  {ticker} pending limit order in flight — skipping")
                return

            if is_on_cooldown(ticker):
                print(f"  {ticker} cooldown {cooldown_remaining(ticker)}s"); return

            # Anti-chop gate: block re-entry if stopped out CHOP_BLOCK_THRESHOLD times today
            today_str = now_et().strftime("%Y-%m-%d")
            stops_today = _daily_stop_counts.get(ticker, {}).get(today_str, 0)
            if stops_today >= CHOP_BLOCK_THRESHOLD:
                print(f"  CHOP BLOCK: {ticker} stopped out {stops_today}x today"); return

            blackout, event_name = cached_blackout_today()
            if blackout:
                print(f"  MACRO BLACKOUT: {event_name} — no trades today"); return

            if not can_enter_new_position():
                return

            # Correlation check — avoid doubling up on highly correlated positions
            if open_positions and is_correlated_with_open_position(ticker):
                return

            # Earnings risk — cached per ticker per day
            today_str = now_et().strftime("%Y-%m-%d")
            cached_earn = _earnings_cache.get(ticker)
            if cached_earn and cached_earn[0] == today_str:
                earn_risk, earn_adj, earn_detail = cached_earn[1]
            else:
                earn_risk, earn_adj, earn_detail = check_earnings_risk(api, ticker)
                _earnings_cache[ticker] = (today_str, (earn_risk, earn_adj, earn_detail))
            if earn_risk == "high":
                print(f"  {ticker} earnings risk HIGH — skipping"); return

            acct_size = get_account_size()
            dt_count = 0
            if acct_size < PDT_ACCOUNT_LIMIT:
                dt_count, _, dt_reset = count_rolling_day_trades()
                if dt_count >= PDT_MAX:
                    print(f"  PDT limit: {dt_count}/{PDT_MAX} day trades in rolling 5-day window"
                          + (f" — resets {dt_reset}" if dt_reset else "")); return

            # Per-ticker day trade gate: one complete round-trip per ticker per day
            today_str = now_et().strftime("%Y-%m-%d")
            ticker_dts = _daily_ticker_dt_counts.get(ticker, {}).get(today_str, 0)
            if ticker_dts >= 1:
                print(f"  {ticker} already day-traded today — skipping re-entry")
                return
            if not passes_filters(ticker, price, acct_size):
                return  # passes_filters already prints its reason

            # Real-time trend check: if price is below the 20-bar MA and prediction
            # isn't strongly bullish, skip the buy. Catches downtrends early without
            # waiting for the next prediction cycle.
            ph = price_history.get(ticker, [])
            if len(ph) >= 10:
                ma20 = sum(ph[-20:]) / min(len(ph), 20)
                pred_score_rt = prediction_cache.get(ticker, {}).get("score", 0)
                if price < ma20 * 0.995 and pred_score_rt < PRED_STRONG_BUY:
                    print(f"  {ticker} below MA20 (${ma20:.2f}) + pred not strong "
                          f"({pred_score_rt:+.0f}) — skipping buy")
                    return

            pred_score  = prediction_cache.get(ticker, {}).get("score", 0)
            rvol        = rvol_cache.get(ticker, 1.0)
            is_gap_play = (ticker in gap_candidates and
                           gap_candidates.get(ticker, {}).get("direction") == "up" and
                           abs(gap_candidates.get(ticker, {}).get("gap_pct", 0)) >= GAP_GO_MIN_PCT)
            qty         = position_size(price, acct_size, pred_score, rvol, is_gap_play)
            if qty == 0:
                print(f"  {ticker} qty 0"); return

            stop   = round(price*(1-STOP_LOSS_PCT), 3)
            target = round(price*(1+TAKE_PROFIT_PCT), 3)
            atr    = None
            try:
                _, calculate_stops = get_predictor()
                if calculate_stops:
                    stop, target, atr = calculate_stops(api, ticker, price,
                                                        STOP_LOSS_PCT, TAKE_PROFIT_PCT)
            except: pass

            if is_gap_play:
                target = round(price * 2.0, 3)   # gap-and-go: target 100% gain (let momentum run)
            if (target-price)/price < MIN_PROFIT_PCT:
                print(f"  {ticker} profit too low"); return

            with _trade_count_lock:
                today = now_et().date()
                if daily_trade_date != today:
                    daily_trade_count = 0; daily_trade_date = today
                if daily_trade_count >= MAX_DAILY_TRADES:
                    print(f"  Daily limit reached (race)")
                    try:
                        _reset_ops_stats_if_new_day()
                        _ops_stats["daily_limit_hits"][ticker] = _ops_stats["daily_limit_hits"].get(ticker, 0) + 1
                        _ops_stats["blocked_buy_tickers"].add(ticker)
                    except Exception:
                        pass
                    return
                # Rolling 2-hour window throttle — spread entries across the trading day
                # so the full daily budget isn't consumed in the first session
                _now_ts = time.time()
                _trade_window_times[:] = [t for t in _trade_window_times
                                          if t >= _now_ts - TRADE_WINDOW_SECS]
                if len(_trade_window_times) >= MAX_WINDOW_TRADES:
                    print(f"  Window limit: {len(_trade_window_times)}/{MAX_WINDOW_TRADES} "
                          f"trades in last {TRADE_WINDOW_SECS//60}min")
                    return
                daily_trade_count += 1
                current_count = daily_trade_count
                _trade_window_times.append(_now_ts)

            # Final race check — roll back count if position appeared concurrently
            with _positions_lock:
                if ticker in open_positions:
                    print(f"  {ticker} position opened concurrently, skipping")
                    with _trade_count_lock:
                        daily_trade_count = max(0, daily_trade_count - 1)
                        try: _trade_window_times.remove(_now_ts)
                        except ValueError: pass
                    return

            # Submit as limit order at 0.3% above current price.
            # Eliminates market-order slippage on thin stocks while still filling
            # quickly (aggressive limit).  Position is registered when the fill
            # confirmation arrives in check_pending_orders().
            limit_price = round(price * 1.003, 2)
            active_sigs = [s["name"] for s in signals if s["action"]=="buy"]
            try:
                order = api.submit_order(
                    symbol=ticker, qty=qty, side="buy",
                    type="limit", limit_price=limit_price,
                    time_in_force="day"
                )
            except Exception as order_err:
                print(f"  Order failed {ticker}: {order_err}")
                with _trade_count_lock:
                    daily_trade_count = max(0, daily_trade_count - 1)
                    try: _trade_window_times.remove(_now_ts)
                    except ValueError: pass
                return

            with _pending_lock:
                _pending_orders[order.id] = {
                    "ticker":        ticker,
                    "qty":           qty,
                    "stop":          stop,
                    "target":        target,
                    "atr":           atr,
                    "active_signals":active_sigs,
                    "price":         price,
                    "pred_score":    pred_score,
                    "rvol":          rvol,
                    "is_gap_play":   is_gap_play,
                    "submitted_at":  time.time(),
                    "window_ts":     _now_ts,
                    "ts":            int(time.time() * 1000),
                }
                _pending_buy_tickers.add(ticker)

            _pdt_cache["ts"] = 0.0  # force refresh so next check sees this order

            mode = "🔴 LIVE" if LIVE_MODE else "PAPER"
            print(f"{mode} LIMIT ORDER {qty}x {ticker} @ ${limit_price:.3f} "
                  f"SL${stop} TP${target} pred={pred_score:+.0f} "
                  f"#{current_count}/{MAX_DAILY_TRADES}"
                  + (f" PDT:{dt_count+1}/{PDT_MAX}" if acct_size < PDT_ACCOUNT_LIMIT else ""))

        elif action == "sell" and ticker in open_positions:
            pos = open_positions.get(ticker, {})
            held_secs = time.time() - pos.get("hold_since", 0)
            if held_secs < 300:
                print(f"  {ticker} min hold: {int(held_secs)}s < 300s — signal sell suppressed")
            else:
                force_sell(ticker, price, reason=reason or "signal")
    except Exception as e:
        print(f"Order error {ticker}: {e}")

# ── Universe / tickers ────────────────────────────────────────

def validate_fallback_tickers():
    global active_tickers
    acct_size = get_account_size()
    floor     = get_price_floor(acct_size)
    ceiling   = get_price_ceiling(acct_size)
    valid     = []
    for ticker in FALLBACK_TICKERS:
        try:
            bar   = api.get_latest_bar(ticker, feed="iex")
            price = float(bar.c)
            if floor <= price <= ceiling:
                valid.append(ticker)
                ph, vh = load_price_history(ticker)
                if ph:
                    price_history[ticker]  = ph
                    volume_history[ticker] = vh
                    print(f"  {ticker} OK @ ${price:.2f} (loaded {len(ph)} bars)")
                else:
                    price_history.setdefault(ticker, [])
                    volume_history.setdefault(ticker, [])
                    print(f"  {ticker} OK @ ${price:.2f}")
            else:
                print(f"  {ticker} out of range @ ${price:.2f}")
        except Exception as e:
            print(f"  {ticker} error: {e}")
    active_tickers = valid if valid else ["SOFI","NIO","MARA","PLTR","DKNG"]
    print(f"Active tickers: {active_tickers}")

def apply_scan_results(results_today, acct_size=None):
    global active_tickers
    if acct_size is None: acct_size = get_account_size()
    affordable = [s for s in results_today
                  if get_price_floor(acct_size)<=s["price"]<=get_price_ceiling(acct_size)
                  and s.get("grade","F") in MIN_GRADE]
    if affordable:
        new_tickers = [s["ticker"] for s in affordable[:8]]
        for s in affordable[:8]:
            ticker_grades[s["ticker"]] = s.get("grade","C")
        for t in new_tickers:
            ph, vh = load_price_history(t)
            price_history[t]  = ph if ph else []
            volume_history[t] = vh if vh else []
        active_tickers = new_tickers
        print(f"Active tickers updated: {active_tickers}")
        try:
            _ops_stats["scan_tickers"] = new_tickers
        except Exception:
            pass
        try:
            scores = {s["ticker"]:s["score"] for s in affordable[:8]}
            save_scan_result(new_tickers, scores)
        except: pass

# NEW: dynamic rvol promotion — swap in high-rvol tickers from seed universe
def promote_rvol_tickers():
    """
    Checks the seed universe every 15 minutes for tickers with exceptional
    rvol (>= RVOL_PROMOTE_MIN). If found and affordable, promotes them into
    the active list, replacing the lowest-scored current ticker.
    """
    global active_tickers
    _, SEED = get_scanner()
    candidates = SEED if SEED else FALLBACK_TICKERS
    acct_size  = get_account_size()
    floor      = get_price_floor(acct_size)
    ceiling    = get_price_ceiling(acct_size)
    promoted   = []

    # Only check tickers NOT already being watched
    check = [t for t in candidates if t not in active_tickers][:30]
    for ticker in check:
        try:
            bar   = api.get_latest_bar(ticker, feed="iex")
            price = float(bar.c)
            if not (floor <= price <= ceiling):
                continue

            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=10)
            bars  = api.get_bars(ticker,"1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10, feed="iex").df
            if bars is None or bars.empty: continue
            if hasattr(bars.index,'levels'):
                if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                else: continue

            avg_vol   = float(bars["volume"].mean())
            now       = now_et()
            mins_open = max(1,(now.hour-9)*60+now.minute-30)
            projected = float(bar.v)*(390/mins_open) if mins_open<390 else float(bar.v)
            rvol      = projected/avg_vol if avg_vol>0 else 1.0

            if rvol >= RVOL_PROMOTE_MIN:
                promoted.append({"ticker":ticker,"price":price,"rvol":round(rvol,2)})
                rvol_cache[ticker] = round(rvol, 2)
                print(f"  PROMOTE {ticker} rvol={rvol:.1f}x @ ${price:.2f}")

            time.sleep(0.15)
        except: continue

    if promoted:
        promoted.sort(key=lambda x: x["rvol"], reverse=True)
        for p in promoted[:2]:   # promote up to 2 tickers at a time
            t = p["ticker"]
            if t not in active_tickers:
                # Replace last ticker in list (lowest priority)
                if len(active_tickers) >= 8:
                    removed = active_tickers[-1]
                    active_tickers = active_tickers[:-1]
                    print(f"  Swapped {removed} → {t} (rvol {p['rvol']}x)")
                active_tickers = [t] + active_tickers
                price_history.setdefault(t, [])
                volume_history.setdefault(t, [])
        print(f"Active tickers after promotion: {active_tickers}")

def run_gap_scan():
    global gap_candidates, active_tickers
    print("=== Gap scan ===")
    _, SEED_UNIVERSE = get_scanner()
    seed = SEED_UNIVERSE if SEED_UNIVERSE else FALLBACK_TICKERS
    acct_size = get_account_size()
    floor     = get_price_floor(acct_size)
    ceiling   = get_price_ceiling(acct_size)
    candidates = {}
    check_list = list(set(seed + list(active_tickers)))
    for ticker in check_list:
        try:
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=5)
            bars  = api.get_bars(ticker,"1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=5, feed="iex").df
            if bars is None or bars.empty or len(bars)<2: continue
            if hasattr(bars.index,'levels'):
                if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                else: continue
            prev_close = float(bars.iloc[-1]["close"])
            avg_vol    = float(bars["volume"].mean())
            latest     = api.get_latest_bar(ticker, feed="iex")
            cur_price  = float(latest.c); cur_vol = float(latest.v)
            if not(floor<=cur_price<=ceiling): continue
            gap_pct = (cur_price-prev_close)/prev_close*100
            rvol    = cur_vol/(avg_vol/390) if avg_vol>0 else 1
            if GAP_MIN_PCT<=abs(gap_pct)<=GAP_MAX_PCT and rvol>=GAP_MIN_RVOL:
                candidates[ticker] = {
                    "ticker":ticker,"price":round(cur_price,2),
                    "prev_close":round(prev_close,2),"gap_pct":round(gap_pct,2),
                    "rvol":round(rvol,2),"direction":"up" if gap_pct>0 else "down",
                }
                print(f"  GAP {ticker} {gap_pct:+.1f}% rvol={rvol:.1f}x")
            time.sleep(0.2)
        except: continue
    # Score each candidate with catalyst tracker (EDGAR batch-cached, very fast)
    _get_cat, _, _ = get_catalyst_tracker()
    if _get_cat:
        for t, c in candidates.items():
            try:
                cat = _get_cat(t)
                c["catalyst_score"]    = cat.get("score", 0)
                c["catalyst_headline"] = cat.get("headline", "")
                c["catalyst_kw"]       = cat.get("kw", "")
                catalyst_cache[t]      = cat
            except Exception as e:
                c["catalyst_score"] = 0
                print(f"  Catalyst error {t}: {e}")

    gap_candidates = candidates
    if candidates:
        up_tickers = [t for t in candidates if candidates[t]["direction"]=="up"]
        # Prioritize: catalyst quality first, then RVOL
        up_tickers.sort(
            key=lambda t: (candidates[t].get("catalyst_score", 0), candidates[t]["rvol"]),
            reverse=True,
        )
        gap_tickers = up_tickers[:3]
        current     = [t for t in active_tickers if t not in gap_tickers]
        active_tickers = (gap_tickers + current)[:5]
        for t in gap_tickers:
            ph, vh = load_price_history(t)
            price_history[t]  = ph if ph else []
            volume_history[t] = vh if vh else []
    socketio.emit("gaps", {"candidates":list(candidates.values()),
                            "scanned_at":now_et().strftime("%I:%M %p ET")})
    print(f"=== Gap scan: {len(candidates)} candidates ===")

def update_rvol():
    for ticker in list(active_tickers):
        try:
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=10)
            bars  = api.get_bars(ticker,"1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10, feed="iex").df
            if bars is None or bars.empty: continue
            if hasattr(bars.index,'levels'):
                if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                else: continue
            avg_vol   = float(bars["volume"].mean())
            latest    = api.get_latest_bar(ticker, feed="iex")
            today_vol = float(latest.v)
            now       = now_et()
            mins_open = max(1,(now.hour-9)*60+now.minute-30)
            # IEX sometimes returns v=0 for stocks that trade on NASDAQ/NYSE,
            # not IEX. Fall back to volume history, but if that's also near-zero
            # it means IEX has no coverage — don't cache anything (gate treats
            # None as unknown and allows the trade through).
            if today_vol <= 0:
                vols = volume_history.get(ticker, [])
                if vols: today_vol = sum(vols[-10:]) * (390 / mins_open / 10)
            if today_vol < 500:
                # No real IEX volume data — leave rvol_cache[ticker] as None
                time.sleep(0.2)
                continue
            projected = today_vol * (390 / mins_open) if mins_open < 390 else today_vol
            rvol      = projected / avg_vol if avg_vol > 0 else 1.0
            rvol_cache[ticker] = round(rvol, 2)
            print(f"  rvol {ticker}: {rvol:.2f}x")
            time.sleep(0.2)
        except Exception as e: print(f"  update_rvol {ticker}: {e}")

# ── Background loops ──────────────────────────────────────────

def macro_loop():
    global macro_status, unusual_volume
    macro_done = set(); macro_day = None
    while True:
        try:
            _hb("macro_loop")
            now  = now_et(); hour = now.hour; day = now.date()
            if day != macro_day: macro_done.clear(); macro_day = day
            if now.weekday()<5 and hour in MACRO_REFRESH_HOURS and hour not in macro_done:
                macro_done.add(hour)
                print("Refreshing macro status...")
                try:
                    status = get_macro_status(api)
                    try:
                        hot = get_hot_sectors(api, top_n=3)
                        status["hot_sectors"] = hot
                    except: status["hot_sectors"] = []
                    macro_status.update(status)
                    _, SEED = get_scanner()
                    scan_universe = SEED if SEED else FALLBACK_TICKERS
                    acct_size = get_account_size()
                    uv = scan_unusual_volume(api, scan_universe, acct_size)
                    unusual_volume = uv
                    status["unusual_volume"] = uv
                    socketio.emit("macro", status)
                    print(f"Macro: blackout={status.get('blackout')} "
                          f"hot_sectors={status.get('hot_sectors')}")
                    if status.get("blackout"):
                        save_alert("WARN",
                            f"MACRO BLACKOUT: {status.get('blackout_event')} — trading suspended")
                except Exception as e:
                    print(f"Macro refresh error: {e}")
        except Exception as e:
            _thread_errors["macro_loop"] = str(e)
            print(f"Macro loop error: {e}")
        time.sleep(60)

def promote_news_tickers():
    """
    Intraday news check — runs every 30 minutes during market hours.
    Pulls market-wide news, finds tickers trending in headlines that aren't
    already being watched, and injects the top 2 into the active list.
    This catches breaking stories (FDA, earnings, M&A, IPOs) between the
    scheduled 10 AM and noon full scans.
    """
    global active_tickers
    try:
        from scanner import get_news_universe
    except Exception as e:
        print(f"  promote_news_tickers import error: {e}")
        return

    acct_size = get_account_size()
    floor     = get_price_floor(acct_size)
    ceiling   = get_price_ceiling(acct_size)

    try:
        news_items = get_news_universe(api)
    except Exception as e:
        print(f"  promote_news_tickers: news fetch error: {e}")
        return

    if not news_items:
        return

    movers = [
        item["symbol"] for item in news_items
        if item["symbol"] not in active_tickers
        and floor <= item.get("price", 0) <= ceiling
    ]

    if not movers:
        return

    print(f"  Intraday news movers: {movers[:5]}")
    for sym in movers[:2]:
        if sym not in active_tickers:
            if len(active_tickers) >= 8:
                removed = active_tickers[-1]
                active_tickers = active_tickers[:-1]
                print(f"  News inject: {sym} replaced {removed}")
            active_tickers = [sym] + active_tickers
            price_history.setdefault(sym, [])
            volume_history.setdefault(sym, [])
    print(f"  Active tickers after news inject: {active_tickers}")


def premarket_loop():
    gap_day = None; rvol_buckets = set(); rvol_day = None
    promote_buckets = set(); promote_day = None
    news_buckets = set(); news_day = None
    while True:
        try:
            _hb("premarket_loop")
            now = now_et(); day = now.date()
            if day != rvol_day:     rvol_buckets.clear();    rvol_day    = day
            if day != promote_day:  promote_buckets.clear(); promote_day = day
            if day != news_day:     news_buckets.clear();    news_day    = day

            # Gap scan: 8:45 AM premarket OR first time in window after a mid-day deploy
            if now.weekday()<5 and gap_day!=day:
                premarket_ready = (now.hour==8 and now.minute>=45)
                inday_first_run = in_trading_window() and is_market_open()
                if premarket_ready or inday_first_run:
                    gap_day = day
                    try: run_gap_scan()
                    except Exception as e: print(f"Gap scan error: {e}")

            if now.weekday()<5 and in_trading_window() and is_market_open():
                bucket = now.hour*4+now.minute//15

                if bucket not in rvol_buckets:
                    rvol_buckets.add(bucket)
                    try: update_rvol(); socketio.emit("rvol", rvol_cache)
                    except Exception as e: print(f"RVOL error: {e}")

                # Dynamic rvol promotion — runs every 15 min same bucket cadence
                if bucket not in promote_buckets:
                    promote_buckets.add(bucket)
                    try: promote_rvol_tickers()
                    except Exception as e: print(f"Promote error: {e}")

                # Intraday news check — every 30 min, catches breaking stories
                # between the scheduled 10 AM and noon full scans
                news_bucket = now.hour*2 + now.minute//30
                if news_bucket not in news_buckets:
                    news_buckets.add(news_bucket)
                    try: promote_news_tickers()
                    except Exception as e: print(f"News inject error: {e}")

        except Exception as e:
            _thread_errors["premarket_loop"] = str(e)
            print(f"Premarket loop error: {e}")
        time.sleep(60)

def prediction_loop():
    global prediction_cache
    pred_done = set(); pred_day = None; last_tickers = []
    while True:
        try:
            _hb("prediction_loop")
            now  = now_et(); hour = now.hour; day = now.date()
            if day != pred_day: pred_done.clear(); pred_day = day
            changed = set(active_tickers) != set(last_tickers)
            should  = (now.weekday()<5 and in_trading_window() and
                      ((hour in PRED_HOURS and hour not in pred_done) or changed))
            if should:
                pred_done.add(hour); last_tickers = list(active_tickers)
                tickers = list(active_tickers)
                print(f"Predictions for {tickers}...")
                try:
                    run_predictions, _ = get_predictor()
                    if run_predictions:
                        results = run_predictions(api, tickers, market_regime)
                        # Remove stale entries for tickers no longer being watched
                        for stale in [t for t in list(prediction_cache) if t not in tickers]:
                            del prediction_cache[stale]
                        prediction_cache.update(results)
                        summary = [(t,f"{r.get('score',0):+.0f}") for t,r in results.items()]
                        print(f"Predictions: {summary}")
                        socketio.emit("predictions", {
                            t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                               "confidence":r.get("confidence","low"),
                               "components":r.get("components",{}),"signals":r.get("signals",[]),
                               "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail",""),
                               "timestamp":r.get("timestamp","")}
                            for t,r in results.items()
                        })
                except Exception as e: print(f"Prediction error: {e}")
        except Exception as e:
            _thread_errors["prediction_loop"] = str(e)
            print(f"Prediction loop error: {e}")
        time.sleep(60)

def scanner_loop():
    global scan_results, active_tickers, market_regime
    scan_done = set(); scan_day = None; startup_scanned = False
    print("Scanner loop started")
    while True:
        try:
            _hb("scanner_loop")
            now  = now_et(); hour = now.hour; day = now.date()
            if day != scan_day:
                scan_done.clear(); scan_day = day; startup_scanned = False
            # Run immediately on the first market-hours iteration so a mid-day
            # deploy doesn't leave active_tickers empty until the next SCAN_HOURS slot.
            need_startup = (not startup_scanned and now.weekday() < 5
                            and is_market_open() and in_trading_window())
            need_hourly  = (now.weekday() < 5 and hour in SCAN_HOURS
                            and hour not in scan_done and in_trading_window())
            if need_startup or need_hourly:
                startup_scanned = True
                if need_hourly: scan_done.add(hour)
                tag = " (startup)" if need_startup and not need_hourly else ""
                print(f"Scanner at {now.strftime('%H:%M')} ET{tag}...")
                try:
                    update_market_regime()
                    acct_size = get_account_size()
                    run_full_scan, _ = get_scanner()
                    if run_full_scan:
                        scan_results = run_full_scan(api)
                        apply_scan_results(scan_results.get("today",[]), acct_size)
                        socketio.emit("scan", scan_results)
                        print(f"Scan done | regime:{market_regime}")
                except Exception as e: print(f"Scanner error: {e}")
        except Exception as e:
            _thread_errors["scanner_loop"] = str(e)
            print(f"Scanner loop error: {e}")
        time.sleep(60)

def db_snapshot_loop():
    global last_db_snapshot
    while True:
        try:
            _hb("db_snapshot_loop")
            now = time.time()
            if now - last_db_snapshot >= DB_SNAPSHOT_INTERVAL:
                last_db_snapshot = now
                try:
                    acct = get_account()
                    if acct:
                        save_portfolio_snapshot(
                            float(acct.equity), float(acct.cash),
                            float(acct.equity)-float(acct.last_equity),
                            market_regime, daily_trade_count
                        )
                    for ticker in list(active_tickers):
                        ph = price_history.get(ticker, [])
                        vh = volume_history.get(ticker, [])
                        if ph: save_price_history(ticker, ph, vh)
                    print("DB snapshot saved")
                except Exception as e:
                    print(f"DB snapshot error: {e}")
        except Exception as e:
            _thread_errors["db_snapshot_loop"] = str(e)
            print(f"DB snapshot loop error: {e}")
        time.sleep(300)

def bot_loop():
    global market_regime
    last_regime    = None
    last_blackout  = None
    eod_closed_date = None   # tracks which day we've already EOD-closed
    hard_close_date = None   # tracks which day the 3:55 failsafe has fired
    while True:
        try:
            _hb("bot_loop")
            _reset_ops_stats_if_new_day()
            now         = now_et()
            market_open = is_market_open()
            window_open = in_trading_window()
            mode        = "🔴LIVE" if LIVE_MODE else "paper"
            print(f"[{mode}] {now.strftime('%H:%M')} ET | "
                  f"mkt={market_open} win={window_open} | "
                  f"{active_tickers} | {market_regime}")

            # EOD hard close — must run BEFORE the window_open check because
            # in_trading_window() returns False at exactly 3:30 PM ET, making
            # any check inside the window block unreachable at that time.
            if now.weekday() < 5 and now.hour == 15 and now.minute >= 28:
                if eod_closed_date != now.date():
                    eod_closed_date = now.date()
                    close_all_positions_eod()

            # Hard failsafe at 3:55 PM — Alpaca-native cancel+close regardless of
            # position tracking state. Fires once per day only.
            if now.weekday() < 5 and now.hour == 15 and now.minute >= 55:
                if hard_close_date != now.date():
                    hard_close_date = now.date()
                    try:
                        api.cancel_all_orders()
                        api.close_all_positions()
                        with _positions_lock:
                            open_positions.clear()
                        print("Hard failsafe: Alpaca cancel+close all fired at 3:55 PM")
                    except Exception as e:
                        print(f"Hard failsafe error: {e}")

            # Blackout check runs BEFORE the window_open guard so that a blackout
            # detected after 3:30 PM (window already closed) still force-closes positions.
            blackout, event_name = cached_blackout_today()
            if blackout:
                if last_blackout != now.strftime("%Y-%m-%d"):
                    close_all_positions_eod()
                    last_blackout = now.strftime("%Y-%m-%d")
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"blackout",
                    "message":f"MACRO BLACKOUT: {event_name} — trading suspended",
                    "live":LIVE_MODE,"regime":market_regime,"blackout":True,
                    "macro":macro_status,"mins_since_open":mins_since_open(),
                    "mins_until_close":mins_until_close()})
                time.sleep(300); continue

            if not window_open:
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Trading window closed — resumes 9:35 AM ET",
                    "live":LIVE_MODE,"regime":market_regime,"blackout":False,
                    "macro":macro_status,"mins_since_open":mins_since_open(),
                    "mins_until_close":mins_until_close()})
                time.sleep(60); continue

            if not market_open:
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Warming up — market opens 9:30 AM ET",
                    "live":LIVE_MODE,"regime":market_regime,"blackout":False,
                    "macro":macro_status,"mins_since_open":mins_since_open(),
                    "mins_until_close":mins_until_close()})
                time.sleep(30); continue

            if not last_regime or (now - last_regime).total_seconds() > 1800:
                update_market_regime(); last_regime = now

            state = {"tickers":{},"account":{},"market_status":"open",
                     "regime":market_regime,"live":LIVE_MODE,
                     "blackout":False,"macro":macro_status,
                     "mins_since_open": mins_since_open(),
                     "mins_until_close": mins_until_close()}

            # Check pending limit orders before processing tickers
            if _pending_orders:
                try: check_pending_orders()
                except Exception as e: print(f"check_pending_orders error: {e}")

            for ticker in list(active_tickers):
                try:
                    bar   = api.get_latest_bar(ticker, feed="iex")
                    price = float(bar.c); vol = float(bar.v)
                    price_history.setdefault(ticker, []).append(price)
                    # Store volume DELTA (not cumulative) — bar.v is cumulative today.
                    prev_vol  = volume_history[ticker][-1] if volume_history.get(ticker) else 0
                    vol_delta = max(0, vol - prev_vol) if prev_vol > 0 else vol
                    volume_history.setdefault(ticker, []).append(vol_delta)
                    if len(price_history[ticker])  > 200: price_history[ticker].pop(0)
                    if len(volume_history[ticker]) > 200: volume_history[ticker].pop(0)
                    # rvol_cache is the single source of truth — populated by update_rvol()
                    # every 15 minutes.  No inline estimate here to avoid formula conflicts.
                    check_stops(ticker, price)
                    sigs              = get_signals(ticker, price)
                    action, reason, buy_w, sell_w = make_decision(ticker, sigs, price)
                    if action != "hold":
                        execute(ticker, action, price, sigs, reason=reason)
                    pos  = open_positions.get(ticker)
                    pred = prediction_cache.get(ticker, {})
                    cd   = cooldown_remaining(ticker)
                    n_b  = sum(1 for s in sigs if s["action"]=="buy")
                    n_s  = sum(1 for s in sigs if s["action"]=="sell")
                    n_v  = sum(1 for s in sigs if s.get("veto"))
                    cat = catalyst_cache.get(ticker, {})
                    state["tickers"][ticker] = {
                        "price":              round(price, 2),
                        "signals":            sigs,
                        "action":             action,
                        "buy_weight":         buy_w,
                        "sell_weight":        sell_w,
                        "stop":               pos["stop"]    if pos else None,
                        "target":             pos["target"]  if pos else None,
                        "partial_done":       pos.get("partial_done", False) if pos else False,
                        "cooldown":           cd,
                        "grade":              ticker_grades.get(ticker, "—"),
                        "pred_score":         pred.get("score",  None),
                        "pred_label":         pred.get("label",  "—"),
                        "pred_conf":          pred.get("confidence","—"),
                        "tf_bias":            pred.get("tf_bias", 0),
                        "rvol":               rvol_cache.get(ticker, None),
                        "is_gap":             ticker in gap_candidates,
                        "veto_count":         n_v,
                        "catalyst_score":     cat.get("score"),
                        "catalyst_headline":  cat.get("headline", ""),
                        "is_gap_play":        pos.get("is_gap_play", False) if pos else False,
                    }
                    print(f"  {ticker}: ${price:.2f} | {action} | "
                          f"v={n_b}b/{n_s}s veto={n_v} w={buy_w:.1f}/{sell_w:.1f} | "
                          f"{market_regime} pred={pred.get('score',0):+.0f}"
                          +(f" rvol={rvol_cache.get(ticker,0):.1f}x" if ticker in rvol_cache else "")
                          +(f" GAP" if ticker in gap_candidates else "")
                          +(f" cd:{cd}s" if cd else ""))
                except Exception as e:
                    print(f"  {ticker} error: {e}")

            state["account"] = get_account_state()
            state["trades"]  = trade_log[:40]
            socketio.emit("state", state)
        except Exception as e:
            _thread_errors["bot_loop"] = str(e)
            print(f"Loop error: {e}")
        time.sleep(INTERVAL)

# ── Socket ────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("Client connected")
    try:
        market_open = is_market_open()
        window_open = in_trading_window()
        both_open   = market_open and window_open

        try:
            db_trades = get_recent_trades(days=14)
            existing_keys = {f"{t['ticker']}-{t.get('entry_ts',0)}" for t in trade_log}
            for t in db_trades:
                key = f"{t['ticker']}-{t.get('entry_ts',0)}"
                if key not in existing_keys:
                    trade_log.append({
                        "type":  "BUY" if t["side"]=="BUY" else "SELL",
                        "ticker":t["ticker"],"qty":t["qty"],
                        "price": t["price"],"pnl":t.get("pnl"),
                        "reason":t.get("reason",""),"ts":t.get("entry_ts",0)
                    })
            trade_log.sort(key=lambda x: x.get("ts",0), reverse=True)
        except Exception as e:
            print(f"DB trade load error: {e}")

        state = {"tickers":{},"account":get_account_state(),
                 "trades":trade_log[:40],
                 "market_status":"open" if both_open else "closed",
                 "live":LIVE_MODE,"macro":macro_status}

        if both_open:
            for ticker in list(active_tickers):
                try:
                    bar   = api.get_latest_bar(ticker, feed="iex")
                    price = float(bar.c)
                    price_history.setdefault(ticker, []).append(price)
                    prev_vol  = volume_history[ticker][-1] if volume_history.get(ticker) else 0
                    vol_delta = max(0, float(bar.v) - prev_vol) if prev_vol > 0 else float(bar.v)
                    volume_history.setdefault(ticker, []).append(vol_delta)
                    sigs  = get_signals(ticker, price)
                    action, reason, buy_w, sell_w = make_decision(ticker, sigs, price)
                    pos   = open_positions.get(ticker)
                    pred  = prediction_cache.get(ticker, {})
                    cat = catalyst_cache.get(ticker, {})
                    state["tickers"][ticker] = {
                        "price":round(price,2),"signals":sigs,
                        "action":action,"buy_weight":buy_w,"sell_weight":sell_w,
                        "stop":pos["stop"]    if pos else None,
                        "target":pos["target"] if pos else None,
                        "partial_done":pos.get("partial_done",False) if pos else False,
                        "cooldown":cooldown_remaining(ticker),
                        "grade":ticker_grades.get(ticker,"—"),
                        "pred_score":pred.get("score",None),
                        "pred_label":pred.get("label","—"),
                        "pred_conf": pred.get("confidence","—"),
                        "tf_bias":pred.get("tf_bias",0),
                        "rvol":rvol_cache.get(ticker,None),
                        "is_gap":ticker in gap_candidates,
                        "veto_count":sum(1 for s in sigs if s.get("veto")),
                        "catalyst_score":    cat.get("score"),
                        "catalyst_headline": cat.get("headline", ""),
                        "is_gap_play":       pos.get("is_gap_play", False) if pos else False,
                    }
                except Exception as e:
                    print(f"on_connect {ticker}: {e}")

        socketio.emit("state", state)
        if scan_results.get("scanned_at"): socketio.emit("scan", scan_results)
        if prediction_cache:
            socketio.emit("predictions", {
                t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                   "confidence":r.get("confidence","low"),
                   "components":r.get("components",{}),"signals":r.get("signals",[]),
                   "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail","")}
                for t,r in prediction_cache.items()
            })
        if macro_status: socketio.emit("macro", macro_status)
        if gap_candidates:
            socketio.emit("gaps", {"candidates":list(gap_candidates.values()),
                                    "scanned_at":now_et().strftime("%I:%M %p ET")})
        if rvol_cache: socketio.emit("rvol", rvol_cache)
        if backtest_cache: socketio.emit("backtest_result", backtest_cache)
        alerts = get_alerts(limit=10, unacknowledged_only=True)
        if alerts: socketio.emit("alerts", alerts)
    except Exception as e:
        print(f"on_connect error: {e}")

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/scanner")
def scanner_page(): return render_template("scanner.html")

@app.route("/predictions")
def predictions_page(): return render_template("predictions.html")

@app.route("/backtest")
def backtest_page(): return render_template("backtest.html")

@app.route("/state")
def state_json(): return jsonify({"trades":trade_log[:20]})

@app.route("/scan")
def scan_json(): return jsonify(scan_results)

@app.route("/predictions/data")
def predictions_json(): return jsonify(prediction_cache)

@app.route("/backtest/data")
def backtest_data_json(): return jsonify(backtest_cache)

@app.route("/gaps")
def gaps_json():
    return jsonify({"candidates":list(gap_candidates.values()),
                    "scanned_at":now_et().strftime("%I:%M %p ET")})

@app.route("/macro")
def macro_json(): return jsonify(macro_status)

@app.route("/alerts")
def alerts_json(): return jsonify(get_alerts(limit=50))

@app.route("/alerts/ack", methods=["POST"])
def ack_alerts():
    acknowledge_alerts()
    return jsonify({"status":"ok"})

@app.route("/history")
def history_json():
    return jsonify({
        "portfolio": get_portfolio_history(days=90),
        "trades":    get_recent_trades(days=30),
        "stats":     get_trade_stats_from_db(),
        "signal_wr": get_signal_win_rates(),
    })

@app.route("/stats")
def stats_json():
    db_stats              = get_trade_stats_from_db()
    blackout, _           = cached_blackout_today()
    pdt_count, pdt_ws, pdt_rd = count_rolling_day_trades()
    return jsonify({
        **db_stats,
        "signal_weights":    signal_weights,
        "signal_win_rates":  get_signal_win_rates(),
        "live_mode":         LIVE_MODE,
        "pdt_rolling_count": pdt_count,
        "pdt_window_start":  pdt_ws,
        "pdt_reset_date":    pdt_rd,
        "daily_trade_count": daily_trade_count,
        "macro_blackout":    blackout,
        "unusual_volume":    unusual_volume[:5],
        "active_tickers":    active_tickers,
        "market_regime":     market_regime,
        "mins_since_open":   mins_since_open(),
        "mins_until_close":  mins_until_close(),
    })

@app.route("/scan/manual", methods=["POST"])
def manual_scan():
    def run():
        global scan_results, active_tickers, market_regime
        try:
            update_market_regime()
            acct_size = get_account_size()
            run_full_scan, _ = get_scanner()
            if not run_full_scan: return
            new_scan = run_full_scan(api)
            def merge(ex, inc):
                seen = {s["ticker"] for s in inc}
                return (inc + [s for s in ex if s["ticker"] not in seen])[:10]
            scan_results["today"]         = merge(scan_results.get("today",[]),     new_scan["today"])
            scan_results["yesterday"]     = merge(scan_results.get("yesterday",[]), new_scan["yesterday"])
            scan_results["scanned_at"]    = new_scan["scanned_at"]
            scan_results["account_size"]  = new_scan.get("account_size", acct_size)
            scan_results["price_range"]   = new_scan.get("price_range","—")
            scan_results["universe_size"]      = new_scan.get("universe_size", 0)
            scan_results["politician_tickers"] = new_scan.get("politician_tickers", [])
            scan_results["manual"]             = True
            apply_scan_results(scan_results["today"], acct_size)
            socketio.emit("scan", scan_results)
        except Exception as e: print(f"Manual scan error: {e}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status":"started"}), 202

@app.route("/predictions/manual", methods=["POST"])
def manual_predictions():
    def run():
        try:
            run_predictions, _ = get_predictor()
            if not run_predictions: return
            results = run_predictions(api, list(active_tickers), market_regime)
            prediction_cache.update(results)
            socketio.emit("predictions", {
                t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                   "confidence":r.get("confidence","low"),
                   "components":r.get("components",{}),"signals":r.get("signals",[]),
                   "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail","")}
                for t,r in results.items()
            })
        except Exception as e: print(f"Manual predictions error: {e}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status":"started"}), 202

@app.route("/backtest/run", methods=["POST"])
def backtest_run():
    def run():
        global backtest_cache
        socketio.emit("backtest_status", {"status":"running","message":"Fetching data..."})
        try:
            run_backtest = get_backtester()
            if not run_backtest:
                socketio.emit("backtest_status", {"status":"error","message":"Not available"})
                return
            result = run_backtest(api, universe_name="both")
            backtest_cache = result
            socketio.emit("backtest_result", result)
            socketio.emit("backtest_status", {"status":"done"})
        except Exception as e:
            print(f"Backtest error: {e}")
            socketio.emit("backtest_status", {"status":"error","message":str(e)})
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status":"started"}), 202

@app.route("/review")
def review_page():
    return render_template("review.html")

@app.route("/review/data")
def review_data():
    try:
        from trade_reviewer import get_trade_review
        trades = get_recent_trades(days=30)
        trade_rev = get_trade_review(trades)

        ops_findings = build_ops_findings(
            _ops_stats, rvol_cache, prediction_cache,
            active_tickers, market_regime
        )

        return jsonify({
            **trade_rev,
            "ops_findings": ops_findings,
            "ops_snapshot": {
                "daily_limit_hits": dict(_ops_stats.get("daily_limit_hits", {})),
                "filter_blocks":    dict(_ops_stats.get("filter_blocks", {})),
                "finbert_method":   _ops_stats.get("finbert_method"),
                "scan_tickers":     _ops_stats.get("scan_tickers", []),
                "blocked_buys":     list(_ops_stats.get("blocked_buy_tickers", set())),
                "active_tickers":   list(active_tickers),
                "regime":           market_regime,
                "rvol_zero_pct":    round(
                    sum(1 for v in rvol_cache.values() if not v) /
                    max(len(rvol_cache), 1) * 100, 1
                ),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "suggestions": [{"priority":"ERROR","area":"System",
                        "finding": str(e), "action":"Check server logs"}]})

@app.route("/review/ai")
def review_ai():
    try:
        from trade_reviewer import get_trade_review
        trades = get_recent_trades(days=7)
        trade_rev = get_trade_review(trades)
        ops_snap = {
            "daily_limit_hits": dict(_ops_stats.get("daily_limit_hits", {})),
            "filter_blocks": dict(_ops_stats.get("filter_blocks", {})),
            "finbert_method": _ops_stats.get("finbert_method"),
            "active_tickers": list(active_tickers),
            "regime": market_regime,
        }
        findings = _ai_analyze(ops_snap, trade_rev.get("summary", {}))
        return jsonify({"findings": findings, "has_key": bool(os.environ.get("ANTHROPIC_API_KEY"))})
    except Exception as e:
        return jsonify({"findings": None, "error": str(e)})

@app.route("/ping")
def ping(): return "pong", 200

@app.route("/health")
def health_page():
    return render_template("health.html")

@app.route("/health/data")
def health_data():
    now_ts = time.time()
    # Max lag (seconds) before each thread is flagged — set generously for infrequent loops
    THREAD_MAX_LAGS = {
        "bot_loop":         INTERVAL * 4,   # runs every 30s
        "premarket_loop":   180,             # runs every 60s
        "macro_loop":       180,             # runs every 60s
        "db_snapshot_loop": 600,             # runs every 300s
        "scanner_loop":     10800,           # runs at 10am/12pm; OK to be idle for hours
        "prediction_loop":  10800,           # runs at 9am/11am; OK to be idle for hours
    }
    threads = {}
    for name, max_lag in THREAD_MAX_LAGS.items():
        last = _thread_heartbeats.get(name)
        lag  = int(now_ts - last) if last else None
        if last is None:
            status = "not_started"
        elif lag < max_lag:
            status = "ok"
        elif lag < max_lag * 2:
            status = "slow"
        else:
            status = "dead"
        threads[name] = {
            "last_beat":  last,
            "lag_secs":   lag,
            "status":     status,
            "last_error": _thread_errors.get(name),
        }

    rvol_vals     = list(rvol_cache.values())
    rvol_zero_pct = round(100 * sum(1 for v in rvol_vals if not v) / max(1, len(rvol_vals)), 1)

    # Per-ticker diagnostics — use real account size for accurate floor/ceiling
    _diag_acct = get_account_size()
    floor      = get_price_floor(_diag_acct)
    ceiling    = get_price_ceiling(_diag_acct)
    ticker_diag = {}
    for t in list(active_tickers):
        ph    = price_history.get(t, [])
        price = ph[-1] if ph else None
        rvol  = rvol_cache.get(t)
        vol   = _vol_check_cache.get(t)
        blocks = []
        if price is not None and not (floor <= price <= ceiling):
            blocks.append(f"price ${price:.2f} outside ${floor:.0f}–${ceiling:.0f}")
        if rvol is not None and 0.10 < rvol < RVOL_THRESHOLD:
            blocks.append(f"rvol {rvol:.1f}x below {RVOL_THRESHOLD}x threshold")
        if vol and not vol[1]:
            blocks.append("avg daily volume too low")
        if is_on_cooldown(t):
            blocks.append(f"cooldown {int(cooldown_remaining(t))}s remaining")
        ticker_diag[t] = {
            "price":         round(price, 2) if price is not None else None,
            "rvol":          round(rvol, 2) if rvol is not None else None,
            "vol_ok":        vol[1] if vol else None,
            "price_depth":   len(ph),
            "blocks":        blocks,
            "passes":        len(blocks) == 0,
            "cooldown_secs": int(cooldown_remaining(t)) if is_on_cooldown(t) else 0,
            "pred_score":    prediction_cache.get(t, {}).get("score"),
            "pred_conf":     prediction_cache.get(t, {}).get("confidence"),
            "has_position":  t in open_positions,
            "pending_order": t in _pending_buy_tickers,
            "grade":         ticker_grades.get(t, "—"),
        }

    blackout, blackout_event  = cached_blackout_today()
    pdt_count, pdt_ws, pdt_rd = count_rolling_day_trades()
    api_lag = int(now_ts - _api_last_success) if _api_last_success else None

    return jsonify({
        "uptime_secs":      int(now_ts - _bot_start_time),
        "threads":          threads,
        "api_lag_secs":     api_lag,
        "api_ok":           api_lag is not None and api_lag < 300,
        "daily_trades":     daily_trade_count,
        "max_daily_trades": MAX_DAILY_TRADES,
        "pdt_rolling_count":pdt_count,
        "pdt_window_start": pdt_ws,
        "pdt_reset_date":   pdt_rd,
        "pdt_safe":         pdt_count < MAX_DAILY_TRADES,
        "open_positions":   len(open_positions),
        "position_tickers": list(open_positions.keys()),
        "pending_orders":   len(_pending_orders),
        "active_tickers":   active_tickers,
        "market_open":      is_market_open(),
        "window_open":      in_trading_window(),
        "blackout":         blackout,
        "blackout_event":   blackout_event,
        "market_regime":    market_regime,
        "rvol_zero_pct":    rvol_zero_pct,
        "rvol_ticker_count":len(rvol_vals),
        "last_scan":        scan_results.get("scanned_at"),
        "live_mode":        LIVE_MODE,
        "ticker_diag":      ticker_diag,
        "signal_weights":   signal_weights,
    })

# ── Startup ───────────────────────────────────────────────────

def reconcile_open_positions():
    """
    Sync open_positions with Alpaca on startup to survive bot restarts.
    Any live position not already in open_positions is restored with fallback
    stop/target so the bot immediately manages its risk.
    """
    try:
        alpaca_positions = api.list_positions()
        if not alpaca_positions:
            print("Reconcile: no open positions in Alpaca")
            return
        # Build a set of tickers bought today so opened_date can be stamped correctly,
        # ensuring the per-ticker day-trade gate still fires after a restart.
        today_str = now_et().strftime("%Y-%m-%d")
        bought_today: set = set()
        try:
            today_dt  = now_et().date()
            midnight  = NY.localize(datetime.combine(today_dt, datetime.min.time()))
            after_str = midnight.strftime("%Y-%m-%dT%H:%M:%SZ")
            todays_orders = api.list_orders(status="filled", after=after_str,
                                            limit=100, direction="desc")
            bought_today = {o.symbol for o in todays_orders if o.side == "buy"}
        except Exception as e:
            print(f"  Reconcile: could not fetch today's orders for opened_date: {e}")
        restored = 0
        for pos in alpaca_positions:
            ticker = pos.symbol
            entry  = float(pos.avg_entry_price)
            qty    = int(pos.qty)
            if qty <= 0:
                continue  # skip short or zero positions; bot only manages longs
            db_stop, db_target = get_open_position_stops(ticker)
            stop   = db_stop   if db_stop   else round(entry * (1 - STOP_LOSS_PCT), 3)
            target = db_target if db_target else round(entry * (1 + TAKE_PROFIT_PCT), 3)
            with _positions_lock:
                if ticker in open_positions:
                    continue  # already tracked; atomic check-then-skip inside lock
                open_positions[ticker] = {
                    "entry": entry, "stop": stop, "target": target,
                    "qty": qty, "atr": None, "active_signals": [],
                    "partial_done": False, "is_gap_play": False,
                    "highest_price": entry,
                    "opened_date": today_str if ticker in bought_today else None,
                    "hold_since": time.time(),
                }
            price_history.setdefault(ticker, [])
            volume_history.setdefault(ticker, [])
            # Add to active_tickers so bot_loop calls check_stops() on every tick.
            # Without this, stops are never enforced for restored positions.
            if ticker not in active_tickers:
                active_tickers.append(ticker)
            restored += 1
            print(f"  Reconcile restored: {qty}x {ticker} @ ${entry:.2f} "
                  f"SL${stop:.3f} TP${target:.3f}")
        if restored:
            print(f"Reconcile: {restored} position(s) restored from Alpaca")
    except Exception as e:
        print(f"Reconcile error: {e}")


def restore_pdt_state():
    """
    On startup: warm the rolling PDT cache and restore today's local trade count.
    The rolling cache is the authoritative gate; the daily counter is a fast
    race-guard and display aid only.
    """
    global daily_trade_count, daily_trade_date
    try:
        count, ws, rd = count_rolling_day_trades()
        print(f"PDT rolling window: {count}/{MAX_DAILY_TRADES} day trades "
              f"(window starts {ws}"
              + (f", resets {rd}" if rd else "") + ")")
        today       = now_et().date()
        midnight_et = NY.localize(datetime.combine(today, datetime.min.time()))
        after_str   = midnight_et.strftime("%Y-%m-%dT%H:%M:%SZ")
        orders      = api.list_orders(status="filled", after=after_str, limit=50, direction="desc")
        daily_trade_count = sum(1 for o in orders if o.side == "buy")
        daily_trade_date  = today
        print(f"Restored daily_trade_count: {daily_trade_count} (today's buys)")
    except Exception as e:
        print(f"restore_pdt_state error: {e}")


def _start_bot():
    port = int(os.environ.get("PORT", 10000))
    print(f"=== BOT v12 | port {port} | {'🔴 LIVE' if LIVE_MODE else '📄 PAPER'} ===")
    print(f"=== Improvements: limit orders | confidence scoring | correlation check | "
          f"social+insider signals | dynamic macro calendar | single RVOL source ===")

    try:
        init_db()
        seed_macro_calendar()
        print("=== Database ready ===")
    except Exception as e:
        print(f"=== DB init error: {e} — continuing without persistence ===")

    update_signal_weights_from_db()

    def startup():
        time.sleep(3)

        if LIVE_MODE:
            try:
                acct   = api.get_account()
                equity = float(acct.equity)
                cash   = float(acct.cash)
                print(f"=== LIVE PRE-FLIGHT: equity=${equity:,.2f}  cash=${cash:,.2f} ===")
                if equity < 5:
                    raise RuntimeError(f"Live account equity ${equity:.2f} is too low to trade safely.")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"Live pre-flight check error: {e}")

        update_market_regime()
        validate_fallback_tickers()
        reconcile_open_positions()
        restore_pdt_state()
        # Cancel any Alpaca orders left open from before this session.
        # On paper accounts, orphaned limit orders accumulate and can fill
        # at stale prices after a restart.
        try:
            api.cancel_all_orders()
            print("Startup: cancelled any leftover open orders from previous session")
        except Exception as e:
            print(f"Startup cancel orders error: {e}")
        # Warm the anti-chop gate from today's filled orders so a mid-day restart
        # doesn't wipe the stop-loss counter and re-open blocked tickers.
        try:
            today_str   = now_et().strftime("%Y-%m-%d")
            midnight_et = NY.localize(datetime.combine(now_et().date(), datetime.min.time()))
            recent_orders = api.list_orders(
                status="filled", after=midnight_et.strftime("%Y-%m-%dT%H:%M:%SZ"),
                limit=100, direction="asc"
            )
            for o in recent_orders:
                if o.side == "sell" and getattr(o, "order_type", "") in ("market", ""):
                    sym = o.symbol
                    _daily_stop_counts.setdefault(sym, {})
                    _daily_stop_counts[sym][today_str] = (
                        _daily_stop_counts[sym].get(today_str, 0) + 1
                    )
            blocked = [s for s, d in _daily_stop_counts.items()
                       if d.get(today_str, 0) >= CHOP_BLOCK_THRESHOLD]
            if blocked:
                print(f"Startup anti-chop: restored stop counts, "
                      f"blocking {blocked} for today")
        except Exception as e:
            print(f"Startup anti-chop restore error: {e}")
        try:
            status = get_macro_status(api)
            status["hot_sectors"] = get_hot_sectors(api)
            macro_status.update(status)
            print(f"Macro: blackout={status.get('blackout')} "
                  f"sectors={status.get('hot_sectors')}")
        except Exception as e:
            print(f"Startup macro error: {e}")
        print(f"=== Ready | tickers={active_tickers} | regime={market_regime} ===")

    threading.Thread(target=startup,          daemon=True).start()
    threading.Thread(target=bot_loop,         daemon=True).start()
    threading.Thread(target=scanner_loop,     daemon=True).start()
    threading.Thread(target=prediction_loop,  daemon=True).start()
    threading.Thread(target=premarket_loop,   daemon=True).start()
    threading.Thread(target=macro_loop,       daemon=True).start()
    threading.Thread(target=db_snapshot_loop, daemon=True).start()

    return port

_port = _start_bot()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=_port)
