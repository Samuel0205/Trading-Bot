import os, time, threading, json
from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# ── Core imports ──────────────────────────────────────────────
from database import (
    init_db, save_trade_open, save_trade_close, get_recent_trades,
    get_trade_stats_from_db, save_signal_performance, get_signal_win_rates,
    save_portfolio_snapshot, get_portfolio_history, save_alert,
    get_alerts, acknowledge_alerts, save_price_history, load_price_history,
    save_scan_result, is_macro_blackout
)
from macro import (
    seed_macro_calendar, check_earnings_risk, is_macro_blackout_today,
    get_sector_momentum, scan_unusual_volume, get_macro_status, get_hot_sectors
)

# ── Lazy imports — prevents startup crash ─────────────────────
def get_scanner():
    try:
        from scanner import run_full_scan, SEED_UNIVERSE
        return run_full_scan, SEED_UNIVERSE
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
# FIX: MAX_ACCOUNT is now a DEFAULT cap only — if account grows beyond it,
# the bot will automatically use the real account size.
# Set MAX_ACCOUNT=30 to start small, but it scales up automatically.
DEFAULT_MAX_ACCOUNT     = float(os.environ.get("MAX_ACCOUNT", "30.00"))
MAX_TRADE_PCT           = 0.45
STOP_LOSS_PCT           = 0.05
TAKE_PROFIT_PCT         = 0.12
INTERVAL                = 30
MIN_PRICE               = 0.50
MIN_VOLUME              = 100_000
MIN_GRADE               = ["A","B","C","D"]
COOLDOWN_STOP           = 900
COOLDOWN_PROFIT         = 600
COOLDOWN_SIGNAL         = 300
TRADING_START_H         = 9
TRADING_START_M         = 35
TRADING_END_H           = 15
TRADING_END_M           = 30
SCAN_HOURS              = [10, 12]
PRED_HOURS              = [9, 11]
MACRO_REFRESH_HOURS     = [8, 12]
MIN_PROFIT_PCT          = 0.04
MAX_DAILY_TRADES        = 3
GAP_MIN_PCT             = 2.0
GAP_MAX_PCT             = 15.0
GAP_MIN_RVOL            = 1.5
RVOL_THRESHOLD          = 1.2
BASE_BUY_THRESHOLD      = 2.0
BASE_SELL_THRESHOLD     = 2.0
PRED_STRONG_BUY         = 40
PRED_SKIP               = -35
PRED_NEED_CONF          = -10
DB_SNAPSHOT_INTERVAL    = 3600

BASE_SIGNAL_WEIGHTS = {
    "MA Crossover":   0.8,
    "RSI":            1.2,
    "Bollinger":      1.0,
    "VWAP":           1.5,
    "MACD":           1.0,
    "Mean Reversion": 0.8,
}

FALLBACK_TICKERS = ["SIRI","TELL","AMC","BB","NOK","MVIS","NIO","MARA","SOFI","CLOV"]

# ── State ─────────────────────────────────────────────────────
price_history     = {}
volume_history    = {}
trade_log         = []
scan_results      = {"today":[],"yesterday":[],"scanned_at":None}
prediction_cache  = {}
backtest_cache    = {}
macro_status      = {}
active_tickers    = list(FALLBACK_TICKERS)
open_positions    = {}
market_regime     = "ranging"
cooldowns         = {}
ticker_grades     = {}
gap_candidates    = {}
rvol_cache        = {}
unusual_volume    = []
daily_trade_count = 0
daily_trade_date  = None
last_db_snapshot  = 0
signal_weights    = dict(BASE_SIGNAL_WEIGHTS)

# FIX: cache blackout result per day — avoids hitting DB on every 30s loop tick
_blackout_cache      = {"date": None, "result": (False, None)}
_blackout_cache_lock = threading.Lock()

# FIX: lock for market_regime to prevent race condition across threads
_regime_lock = threading.Lock()

# FIX: lock for daily_trade_count
_trade_count_lock = threading.Lock()

NY = pytz.timezone("America/New_York")

# ── Blackout cache (eliminates double DB hit) ─────────────────

def cached_blackout_today():
    """
    Returns (is_blackout, event_name).
    Caches result for the current date — safe to call every tick.
    FIX: original called is_macro_blackout_today() in both bot_loop and execute(),
    hitting the DB twice per ticker per 30s cycle. Now cached per day.
    """
    today = datetime.now(NY).strftime("%Y-%m-%d")
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
    try:
        return api.get_account()
    except Exception as e:
        print(f"get_account error: {e}")
        return None

def get_account_size():
    """
    FIX: Original used min(equity, MAX_ACCOUNT) which permanently capped
    growth at $30. Now returns actual equity — DEFAULT_MAX_ACCOUNT is only
    used as a fallback when the API call fails.
    Bot position sizing already scales via MAX_TRADE_PCT, so no hard cap needed.
    """
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
            return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime,"live":LIVE_MODE}
        return {
            "portfolio": round(float(acct.equity), 2),
            "cash":      round(float(acct.cash), 2),
            "pnl":       round(float(acct.equity) - float(acct.last_equity), 2),
            "regime":    market_regime,
            "live":      LIVE_MODE,
        }
    except Exception as e:
        print(f"get_account_state error: {e}")
        return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime,"live":LIVE_MODE}

# ── Scaling ───────────────────────────────────────────────────

def get_price_ceiling(account_size=None):
    if account_size is None: account_size = get_account_size()
    return max(min(account_size * 0.45, 10.00), 0.60)

def get_price_floor(account_size=None):
    if account_size is None: account_size = get_account_size()
    if account_size > 2000: return 5.00
    if account_size > 500:  return 2.00
    if account_size > 100:  return 1.00
    return 0.50

def get_min_volume(account_size=None):
    if account_size is None: account_size = get_account_size()
    if account_size > 5000: return 1_000_000
    if account_size > 1000: return 500_000
    if account_size > 200:  return 250_000
    return 100_000

# ── Time helpers ──────────────────────────────────────────────

def in_trading_window():
    now  = datetime.now(NY)
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return (TRADING_START_H*60+TRADING_START_M) <= mins < (TRADING_END_H*60+TRADING_END_M)

def is_market_open():
    try:
        return api.get_clock().is_open
    except Exception as e:
        print(f"is_market_open error: {e}")
        return False

# ── PDT compliance ────────────────────────────────────────────

def count_day_trades_this_week():
    try:
        now    = datetime.now(NY)
        cutoff = now - timedelta(days=7)
        orders = api.list_orders(
            status="filled",
            after=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=100, direction="desc"
        )
        buys = {}; sells = {}
        for o in orders:
            d = o.filled_at[:10] if o.filled_at else ""
            k = (o.symbol, d)
            if o.side == "buy":  buys[k]  = buys.get(k, 0) + 1
            if o.side == "sell": sells[k] = sells.get(k, 0) + 1
        return sum(1 for k in buys if k in sells)
    except Exception as e:
        print(f"PDT count error: {e}"); return 0

def is_day_trade_safe():
    if not LIVE_MODE: return True
    return count_day_trades_this_week() < MAX_DAILY_TRADES

# ── Cooldowns ─────────────────────────────────────────────────

def set_cooldown(ticker, reason="signal"):
    d = {"stop_loss":COOLDOWN_STOP,"take_profit":COOLDOWN_PROFIT,
         "signal":COOLDOWN_SIGNAL,"eod_close":COOLDOWN_SIGNAL}
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
        return [{"name":k,"action":"hold","signal":50} for k in signal_weights]
    rsi  = calc_rsi(hist)
    ma50 = calc_ma(hist, min(50, n))
    ma200= calc_ma(hist, min(200, n))
    mean, upper, lower = calc_bollinger(hist)
    vwap = calc_vwap(hist[-78:], vols[-78:])
    macd_line = calc_macd(hist)
    z    = (price-mean)/max((upper-mean), 0.01)
    ok_buy  = market_regime in ("trending_up","ranging")
    ok_sell = market_regime in ("trending_down","ranging")
    ma_conf = min(1.0, n/20)

    def act(bc, sc):
        if bc and ok_buy:  return "buy"
        if sc and ok_sell: return "sell"
        return "hold"

    return [
        {"name":"MA Crossover",
         "action":act(ma50>ma200*1.005,ma200>ma50*1.005) if ma_conf>0.5 else "hold",
         "signal":min(95,50+(ma50-ma200)/max(ma200,1)*600)},
        {"name":"RSI","action":act(rsi<35,rsi>65),"signal":100-rsi},
        {"name":"Bollinger","action":act(price<lower*0.99,price>upper*1.01),
         "signal":max(5,min(95,50-z*40))},
        {"name":"VWAP","action":act(price>vwap*1.001,price<vwap*0.999),
         "signal":min(95,max(5,50+(price-vwap)/max(vwap,1)*500))},
        {"name":"MACD","action":act(macd_line>0,macd_line<0),
         "signal":min(95,max(5,50+macd_line*10))},
        {"name":"Mean Reversion","action":act(price<mean*0.96,price>mean*1.04),
         "signal":min(92,max(8,50+(mean-price)/max(mean,1)*250))},
    ]

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
    RTICKERS = ["SIRI","TELL","NIO","MARA","SOFI","AMC","BB","NOK"]
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=15)
        ss    = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        es    = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        closes = None
        for ticker in RTICKERS:
            try:
                bars = api.get_bars(ticker,"1Day",start=ss,end=es,limit=12,feed="iex").df
                if bars is None or bars.empty: continue
                if hasattr(bars.index,'levels'):
                    if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                    else: continue
                if len(bars)>=5:
                    closes=list(bars["close"]); break
            except: continue
        if closes is None:
            all_p=[p for t in active_tickers for p in price_history.get(t,[])[-10:]]
            closes=all_p if len(all_p)>=5 else None
        if closes is None:
            print(f"Regime: no data, keeping {market_regime}"); return
        ma5  = calc_ma(closes, min(5,  len(closes)))
        ma10 = calc_ma(closes, min(10, len(closes)))
        latest = closes[-1]
        # FIX: use lock when writing shared global
        with _regime_lock:
            if   ma5>ma10*1.005 and latest>ma5: market_regime="trending_up"
            elif ma5<ma10*0.995 and latest<ma5: market_regime="trending_down"
            else:                               market_regime="ranging"
        print(f"Regime: {market_regime}")
    except Exception as e:
        print(f"Regime error: {e}")

# ── Filters ───────────────────────────────────────────────────

def passes_filters(ticker, price, account_size=None):
    if account_size is None: account_size = get_account_size()
    floor   = get_price_floor(account_size)
    ceiling = get_price_ceiling(account_size)
    min_vol = get_min_volume(account_size)
    if not(floor<=price<=ceiling):
        print(f"  {ticker} ${price:.2f} outside range ${floor}–${ceiling}"); return False
    grade = ticker_grades.get(ticker)
    if grade and grade not in MIN_GRADE:
        print(f"  {ticker} grade {grade} excluded"); return False
    rvol = rvol_cache.get(ticker)
    if rvol is not None and rvol < RVOL_THRESHOLD:
        print(f"  {ticker} rvol {rvol:.2f}x low"); return False
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=5)
        bars  = api.get_bars(ticker,"1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=5, feed="iex").df
        if bars is None or bars.empty: return False
        if hasattr(bars.index,'levels'):
            if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
            else: return False
        if float(bars["volume"].mean()) < min_vol:
            print(f"  {ticker} low volume"); return False
        return True
    except Exception as e:
        print(f"  Filter error {ticker}: {e}"); return False

# ── Position sizing ───────────────────────────────────────────

def position_size(price, account_size=None, pred_score=0, rvol=1.0):
    try:
        if account_size is None: account_size = get_account_size()
        usable = get_available_cash()
        if usable <= 0: return 0
        if   pred_score >= PRED_STRONG_BUY: pct = 0.60
        elif pred_score >= 0:               pct = MAX_TRADE_PCT
        else:                               pct = 0.30
        if   rvol >= 3.0: pct = min(pct*1.15, 0.65)
        elif rvol >= 2.0: pct = min(pct*1.05, 0.55)
        if   price < 1.00: pct *= 0.80
        elif price < 2.00: pct *= 0.90
        max_spend = usable * pct
        if max_spend < price:
            print(f"  Can't afford ${price:.2f} (max ${max_spend:.2f})"); return 0
        return max(0, int(max_spend/price))
    except Exception as e:
        print(f"  position_size error: {e}"); return 0

# ── Stops ─────────────────────────────────────────────────────

def check_stops(ticker, price):
    pos = open_positions.get(ticker)
    if not pos: return
    if price > pos["entry"]*1.05:
        new_stop = max(pos["stop"], pos["entry"]*1.01)
        if new_stop > pos["stop"]:
            open_positions[ticker]["stop"] = round(new_stop, 3)
            print(f"  Trail stop {ticker} → ${new_stop:.3f}")
    if   price <= pos["stop"]:   force_sell(ticker, price, reason="stop_loss")
    elif price >= pos["target"]: force_sell(ticker, price, reason="take_profit")

def force_sell(ticker, price, reason="stop_loss"):
    try:
        pos_api = api.get_position(ticker)
        qty     = int(pos_api.qty)
        if qty <= 0:
            open_positions.pop(ticker, None); return
        entry   = float(pos_api.avg_entry_price)
        pnl     = round((price - entry) * qty, 2)
        api.submit_order(symbol=ticker, qty=qty, side="sell",
                         type="market", time_in_force="day")
        was_win  = pnl > 0
        pos_data = open_positions.get(ticker, {})
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
        open_positions.pop(ticker, None)
        set_cooldown(ticker, reason)

        wr   = get_trade_stats_from_db().get("win_rate", 0)
        mode = "🔴 LIVE" if LIVE_MODE else "PAPER"
        print(f"{mode} SELL {qty}x {ticker} @ ${price:.2f} "
              f"| {reason} | PnL ${pnl:+.2f} | DB WR:{wr:.0f}%")

        if LIVE_MODE and abs(pnl) > 1:
            save_alert("INFO" if was_win else "WARN",
                       f"{'WIN' if was_win else 'LOSS'} ${pnl:+.2f} on {ticker} ({reason})",
                       ticker)
    except Exception as e:
        print(f"Force sell error {ticker}: {e}")

def close_all_positions_eod():
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

    if pscore <= PRED_SKIP:
        return "hold", f"pred_skip({pscore})", buy_w, sell_w

    if   pscore >= PRED_STRONG_BUY: bt = st = 1.2
    elif pscore >= 20:              bt = st = 1.5
    elif pscore < PRED_NEED_CONF:   bt = 2.5; st = 2.0
    else:                           bt = BASE_BUY_THRESHOLD; st = BASE_SELL_THRESHOLD

    if is_gap and gap_candidates.get(ticker, {}).get("direction") == "up":
        bt = max(1.0, bt-0.3)
    if rvol >= 2.0:
        bt = max(1.0, bt-0.2); st = max(1.0, st-0.2)
    if tf_bias == -1:
        bt += 0.8

    # Sector alignment bonus
    hot_sectors = macro_status.get("hot_sectors", [])
    if hot_sectors:
        bt = max(1.0, bt-0.1)

    if   buy_w  >= bt: action = "buy"
    elif sell_w >= st: action = "sell"
    else:              action = "hold"

    reason = (f"bw={buy_w:.1f}>={bt:.1f}" if action=="buy"
              else f"sw={sell_w:.1f}>={st:.1f}" if action=="sell"
              else f"hold(b={buy_w:.1f},s={sell_w:.1f})")
    return action, reason, buy_w, sell_w

# ── Trade execution ───────────────────────────────────────────

def execute(ticker, action, price, signals, reason="signal"):
    global daily_trade_count, daily_trade_date
    try:
        if action == "buy" and ticker not in open_positions:
            if is_on_cooldown(ticker):
                print(f"  {ticker} cooldown {cooldown_remaining(ticker)}s"); return

            # FIX: use cached blackout — no DB hit per tick
            blackout, event_name = cached_blackout_today()
            if blackout:
                print(f"  MACRO BLACKOUT: {event_name} — no trades today"); return

            earn_risk, earn_adj, earn_detail = check_earnings_risk(api, ticker)
            if earn_risk == "high":
                print(f"  {ticker} earnings risk HIGH — skipping"); return

            if LIVE_MODE and not is_day_trade_safe():
                print(f"  PDT limit — skipping"); return

            today = datetime.now(NY).date()
            # Check limit BEFORE doing expensive passes_filters/qty work
            with _trade_count_lock:
                if daily_trade_date != today:
                    daily_trade_count = 0; daily_trade_date = today
                if daily_trade_count >= MAX_DAILY_TRADES:
                    print(f"  Daily limit {MAX_DAILY_TRADES} reached"); return

            acct_size = get_account_size()
            if not passes_filters(ticker, price, acct_size):
                return

            pred_score = prediction_cache.get(ticker, {}).get("score", 0)
            rvol       = rvol_cache.get(ticker, 1.0)
            qty        = position_size(price, acct_size, pred_score, rvol)
            if qty == 0:
                print(f"  {ticker} qty 0")
                return

            stop   = round(price*(1-STOP_LOSS_PCT), 3)
            target = round(price*(1+TAKE_PROFIT_PCT), 3)
            atr    = None
            try:
                _, calculate_stops = get_predictor()
                if calculate_stops:
                    stop, target, atr = calculate_stops(api, ticker, price,
                                                        STOP_LOSS_PCT, TAKE_PROFIT_PCT)
            except: pass

            if (target-price)/price < MIN_PROFIT_PCT:
                print(f"  {ticker} profit too low")
                return

            # FIX: increment ONLY when we're actually about to submit the order
            # No rollback needed — if submit_order fails, we catch the exception below
            with _trade_count_lock:
                if daily_trade_count >= MAX_DAILY_TRADES:
                    print(f"  Daily limit reached (race)"); return
                daily_trade_count += 1
                current_count = daily_trade_count

            api.submit_order(symbol=ticker, qty=qty, side="buy",
                             type="market", time_in_force="day")

            active_sigs = [s["name"] for s in signals if s["action"]=="buy"]
            ts = int(time.time()*1000)
            open_positions[ticker] = {
                "entry":price,"stop":stop,"target":target,
                "qty":qty,"atr":atr,"active_signals":active_sigs
            }

            try:
                save_trade_open(ticker, qty, price, stop, target, pred_score,
                                rvol, ticker in gap_candidates, active_sigs, ts)
            except Exception as e:
                print(f"  DB save_trade_open error: {e}")

            rr = round((target-price)/(price-stop), 2) if price > stop else "?"
            trade_log.insert(0, {
                "type":"BUY","ticker":ticker,"qty":qty,
                "price":round(price,2),"pnl":None,
                "stop":stop,"target":target,
                "pred_score":pred_score,"rvol":rvol,
                "gap":ticker in gap_candidates,"reason":reason,
                "ts":ts
            })
            mode = "🔴 LIVE" if LIVE_MODE else "PAPER"
            print(f"{mode} BUY {qty}x {ticker} @ ${price:.2f} "
                  f"SL${stop} TP${target} R:R={rr} "
                  f"pred={pred_score:+.0f} rvol={rvol:.1f}x "
                  f"#{current_count}/{MAX_DAILY_TRADES}")
            if LIVE_MODE:
                save_alert("INFO",
                    f"BUY {qty}x {ticker} @ ${price:.2f} | pred={pred_score:+.0f}", ticker)

        elif action == "sell" and ticker in open_positions:
            force_sell(ticker, price, reason=reason or "signal")
    except Exception as e:
        print(f"Order error {ticker}: {e}")

# ── Fallback + scan ───────────────────────────────────────────

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
    active_tickers = valid if valid else ["SIRI","TELL","AMC","BB","NOK"]
    print(f"Active tickers: {active_tickers}")

def apply_scan_results(results_today, acct_size=None):
    global active_tickers
    if acct_size is None: acct_size = get_account_size()
    affordable = [s for s in results_today
                  if get_price_floor(acct_size)<=s["price"]<=get_price_ceiling(acct_size)
                  and s.get("grade","F") in MIN_GRADE]
    if affordable:
        new_tickers = [s["ticker"] for s in affordable[:5]]
        for s in affordable[:5]:
            ticker_grades[s["ticker"]] = s.get("grade","C")
        for t in new_tickers:
            ph, vh = load_price_history(t)
            price_history[t]  = ph if ph else []
            volume_history[t] = vh if vh else []
        active_tickers = new_tickers
        print(f"Active tickers updated: {active_tickers}")
        try:
            scores = {s["ticker"]:s["score"] for s in affordable[:5]}
            save_scan_result(new_tickers, scores)
        except: pass

# ── Gap scanner ───────────────────────────────────────────────

def run_gap_scan():
    global gap_candidates, active_tickers
    print("=== Gap scan ===")
    _, SEED_UNIVERSE = get_scanner()
    # FIX: fallback to FALLBACK_TICKERS if scanner import failed or returned empty
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
    gap_candidates = candidates
    if candidates:
        gap_tickers = [t for t in candidates if candidates[t]["direction"]=="up"][:3]
        current     = [t for t in active_tickers if t not in gap_tickers]
        active_tickers = (gap_tickers + current)[:5]
        for t in gap_tickers:
            ph, vh = load_price_history(t)
            price_history[t]  = ph if ph else []
            volume_history[t] = vh if vh else []
    socketio.emit("gaps", {"candidates":list(candidates.values()),
                            "scanned_at":datetime.now(NY).strftime("%I:%M %p ET")})
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
            now       = datetime.now(NY)
            mins_open = max(1,(now.hour-9)*60+now.minute-30)
            projected = today_vol*(390/mins_open) if mins_open<390 else today_vol
            rvol      = projected/avg_vol if avg_vol>0 else 1.0
            rvol_cache[ticker] = round(rvol, 2)
            time.sleep(0.2)
        except: pass

# ── Background loops ──────────────────────────────────────────

def macro_loop():
    global macro_status, unusual_volume
    macro_done = set(); macro_day = None
    while True:
        try:
            now  = datetime.now(NY); hour = now.hour; day = now.date()
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
                    # FIX: fallback so unusual volume scan always has something to check
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
            print(f"Macro loop error: {e}")
        time.sleep(60)

def premarket_loop():
    gap_day = None; rvol_buckets = set(); rvol_day = None
    while True:
        try:
            now = datetime.now(NY); day = now.date()
            if day != rvol_day: rvol_buckets.clear(); rvol_day = day
            if now.weekday()<5 and now.hour==8 and now.minute>=45 and gap_day!=day:
                gap_day = day
                try: run_gap_scan()
                except Exception as e: print(f"Gap scan error: {e}")
            if now.weekday()<5 and in_trading_window() and is_market_open():
                bucket = now.hour*4+now.minute//15
                if bucket not in rvol_buckets:
                    rvol_buckets.add(bucket)
                    try: update_rvol(); socketio.emit("rvol", rvol_cache)
                    except Exception as e: print(f"RVOL error: {e}")
        except Exception as e: print(f"Premarket loop error: {e}")
        time.sleep(60)

def prediction_loop():
    global prediction_cache
    pred_done = set(); pred_day = None; last_tickers = []
    while True:
        try:
            now  = datetime.now(NY); hour = now.hour; day = now.date()
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
        except Exception as e: print(f"Prediction loop error: {e}")
        time.sleep(60)

def scanner_loop():
    global scan_results, active_tickers, market_regime
    scan_done = set(); scan_day = None
    print("Scanner loop started")
    while True:
        try:
            now  = datetime.now(NY); hour = now.hour; day = now.date()
            if day != scan_day: scan_done.clear(); scan_day = day
            if (now.weekday()<5 and hour in SCAN_HOURS
                    and hour not in scan_done and in_trading_window()):
                scan_done.add(hour)
                print(f"Scanner at {now.strftime('%H:%M')} ET...")
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
        except Exception as e: print(f"Scanner loop error: {e}")
        time.sleep(60)

def db_snapshot_loop():
    global last_db_snapshot
    while True:
        try:
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
            print(f"DB snapshot loop error: {e}")
        time.sleep(300)

def bot_loop():
    global market_regime
    last_regime    = None
    last_blackout  = None   # FIX: only close positions once on blackout, not every tick
    while True:
        try:
            now          = datetime.now(NY)
            market_open  = is_market_open()
            window_open  = in_trading_window()
            mode         = "🔴LIVE" if LIVE_MODE else "paper"
            print(f"[{mode}] {now.strftime('%H:%M')} ET | "
                  f"mkt={market_open} win={window_open} | "
                  f"{active_tickers} | {market_regime}")

            if not window_open:
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Trading window closed — resumes 9:35 AM ET","live":LIVE_MODE})
                time.sleep(60); continue

            if not market_open:
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Warming up — market opens 9:30 AM ET","live":LIVE_MODE})
                time.sleep(30); continue

            # FIX: use cached blackout — single DB hit per day
            blackout, event_name = cached_blackout_today()
            if blackout:
                if last_blackout != now.strftime("%Y-%m-%d"):
                    close_all_positions_eod()
                    last_blackout = now.strftime("%Y-%m-%d")
                socketio.emit("state", {"tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"blackout",
                    "message":f"MACRO BLACKOUT: {event_name} — trading suspended","live":LIVE_MODE})
                time.sleep(300); continue

            if not last_regime or (now - last_regime).total_seconds() > 1800:
                update_market_regime(); last_regime = now

            if now.hour == 15 and now.minute >= 30:
                close_all_positions_eod()
                time.sleep(600); continue

            state = {"tickers":{},"account":{},"market_status":"open",
                     "regime":market_regime,"live":LIVE_MODE,
                     "blackout":False,"macro":macro_status}

            for ticker in list(active_tickers):
                try:
                    bar   = api.get_latest_bar(ticker, feed="iex")
                    price = float(bar.c); vol = float(bar.v)
                    price_history.setdefault(ticker, []).append(price)
                    volume_history.setdefault(ticker, []).append(vol)
                    if len(price_history[ticker])  > 200: price_history[ticker].pop(0)
                    if len(volume_history[ticker]) > 200: volume_history[ticker].pop(0)
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
                    state["tickers"][ticker] = {
                        "price":      round(price, 2),
                        "signals":    sigs,
                        "action":     action,
                        "buy_weight": buy_w,
                        "sell_weight":sell_w,
                        "stop":       pos["stop"]   if pos else None,
                        "target":     pos["target"] if pos else None,
                        "cooldown":   cd,
                        "grade":      ticker_grades.get(ticker, "—"),
                        "pred_score": pred.get("score",   None),
                        "pred_label": pred.get("label",   "—"),
                        "pred_conf":  pred.get("confidence","—"),
                        "tf_bias":    pred.get("tf_bias",  0),
                        "rvol":       rvol_cache.get(ticker, None),
                        "is_gap":     ticker in gap_candidates,
                    }
                    print(f"  {ticker}: ${price:.2f} | {action} | "
                          f"v={n_b}b/{n_s}s w={buy_w:.1f}/{sell_w:.1f} | "
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
                    volume_history.setdefault(ticker, []).append(float(bar.v))
                    sigs  = get_signals(ticker, price)
                    action, reason, buy_w, sell_w = make_decision(ticker, sigs, price)
                    pos   = open_positions.get(ticker)
                    pred  = prediction_cache.get(ticker, {})
                    state["tickers"][ticker] = {
                        "price":round(price,2),"signals":sigs,
                        "action":action,"buy_weight":buy_w,"sell_weight":sell_w,
                        "stop":pos["stop"]   if pos else None,
                        "target":pos["target"] if pos else None,
                        "cooldown":cooldown_remaining(ticker),
                        "grade":ticker_grades.get(ticker,"—"),
                        "pred_score":pred.get("score",None),
                        "pred_label":pred.get("label","—"),
                        "tf_bias":pred.get("tf_bias",0),
                        "rvol":rvol_cache.get(ticker,None),
                        "is_gap":ticker in gap_candidates,
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
                                    "scanned_at":datetime.now(NY).strftime("%I:%M %p ET")})
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
                    "scanned_at":datetime.now(NY).strftime("%I:%M %p ET")})

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
    db_stats = get_trade_stats_from_db()
    blackout, _ = cached_blackout_today()
    return jsonify({
        **db_stats,
        "signal_weights":       signal_weights,
        "signal_win_rates":     get_signal_win_rates(),
        "live_mode":            LIVE_MODE,
        "pdt_trades_this_week": count_day_trades_this_week() if LIVE_MODE else 0,
        "daily_trade_count":    daily_trade_count,
        "macro_blackout":       blackout,
        "unusual_volume":       unusual_volume[:5],
        "active_tickers":       active_tickers,
        "market_regime":        market_regime,
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
            scan_results["universe_size"] = new_scan.get("universe_size", 0)
            scan_results["manual"]        = True
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

@app.route("/ping")
def ping(): return "pong", 200

# ── Startup ───────────────────────────────────────────────────


# ── Startup — runs regardless of how app is launched ─────────
# Works with: python app.py  OR  gunicorn with eventlet worker
# Threads are started at module load time. socketio.run() is only
# called when launched directly (gunicorn handles serving instead).

def _start_bot():
    """Initialize DB, seed calendar, launch all background threads."""
    port = int(os.environ.get("PORT", 10000))
    print(f"=== BOT v10 | port {port} | {'🔴 LIVE' if LIVE_MODE else '📄 PAPER'} ===")
    print(f"=== DEFAULT_MAX=${DEFAULT_MAX_ACCOUNT} | PDT={MAX_DAILY_TRADES} | "
          f"WINDOW={TRADING_START_H}:{TRADING_START_M:02d}–{TRADING_END_H}:{TRADING_END_M:02d} ET ===")

    try:
        init_db()
        seed_macro_calendar()
        print("=== Database ready ===")
    except Exception as e:
        print(f"=== DB init error: {e} — continuing without persistence ===")

    update_signal_weights_from_db()

    def startup():
        time.sleep(3)
        print("=== Startup: regime ===")
        update_market_regime()
        print("=== Startup: fallback tickers ===")
        validate_fallback_tickers()
        print("=== Startup: macro status ===")
        try:
            status = get_macro_status(api)
            status["hot_sectors"] = get_hot_sectors(api)
            macro_status.update(status)
            print(f"Macro: blackout={status.get('blackout')} sectors={status.get('hot_sectors')}")
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

# Start everything at import time (works for gunicorn AND python app.py)
_port = _start_bot()

if __name__ == "__main__":
    # Direct launch: python app.py
    socketio.run(app, host="0.0.0.0", port=_port)
