import os, time, threading
from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from scanner import run_full_scan
from predictions import run_predictions, calculate_stops

API_KEY    = os.environ.get("APCA_API_KEY_ID")
SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys.")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Config ────────────────────────────────────────────────────
MAX_ACCOUNT      = 20.00
MAX_TRADE_PCT    = 0.50
STOP_LOSS_PCT    = 0.05
TAKE_PROFIT_PCT  = 0.10
INTERVAL         = 15
MIN_PRICE        = 0.50
MIN_VOLUME       = 100_000
MIN_GRADE        = ["A","B","C","D"]
COOLDOWN_STOP    = 900              # 15 min after stop loss
COOLDOWN_PROFIT  = 600              # 10 min after take profit (was 3 min — too short)
COOLDOWN_SIGNAL  = 300              # 5 min after signal exit
MIN_PROFIT_PCT   = 0.03             # minimum 3% expected profit before entering
MAX_DAILY_TRADES = 20               # hard limit — prevent churning
TRADING_START_H  = 8
TRADING_START_M  = 45
TRADING_END_H    = 14
TRADING_END_M    = 0
SCAN_HOURS       = [9, 11, 13]
PRED_HOURS       = [9, 10, 12]
REL_VOL_HOURS    = [9, 10, 11, 12, 13]   # relative volume refresh

# Gap scanner config
GAP_MIN_PCT      = 2.0    # minimum gap % to qualify
GAP_MAX_PCT      = 15.0   # above this = too risky (earnings gap)
GAP_MIN_RVOL     = 1.5    # gap must have elevated relative volume

# Relative volume threshold — stock must be trading above this
# to be considered "in play" for trading
RVOL_THRESHOLD   = 1.3

# Signal weights — tuned by win rate feedback
# Base weights, adjusted dynamically by performance
BASE_SIGNAL_WEIGHTS = {
    "MA Crossover":   0.8,
    "RSI":            1.2,
    "Bollinger":      1.0,
    "VWAP":           1.5,
    "MACD":           1.0,
    "Mean Reversion": 0.8,
}

PRED_STRONG_BUY  =  40
PRED_NORMAL_BUY  =   0
PRED_REDUCE      = -15
PRED_SKIP        = -35
PRED_NEED_CONF   = -10
BASE_BUY_THRESHOLD  = 2.2   # raised from 1.8 — needs stronger agreement
BASE_SELL_THRESHOLD = 2.2

FALLBACK_TICKERS = ["SIRI","TELL","AMC","BB","NOK","MVIS","NIO","MARA","SOFI","CLOV"]

# ── State ─────────────────────────────────────────────────────
price_history    = {}
volume_history   = {}
trade_log        = []
scan_results     = {"today":[],"yesterday":[],"scanned_at":None}
prediction_cache = {}
active_tickers   = list(FALLBACK_TICKERS)
open_positions   = {}
market_regime    = "ranging"
cooldowns        = {}
ticker_grades    = {}
gap_candidates   = {}   # { ticker: gap_data } — pre-market gap picks
rvol_cache       = {}   # { ticker: rvol } — real-time relative volume

# ── Win rate tracking — per signal ────────────────────────────
# Tracks which signals were active on winning vs losing trades
# Used to dynamically adjust SIGNAL_WEIGHTS
signal_performance = {
    name: {"wins": 0, "losses": 0} for name in BASE_SIGNAL_WEIGHTS
}
trade_stats = {
    "wins": 0, "losses": 0, "total": 0,
    "total_pnl": 0.0,
    "best_trade": 0.0,
    "worst_trade": 0.0,
}

# Dynamic weights — start at base, evolve with performance
signal_weights = dict(BASE_SIGNAL_WEIGHTS)

daily_trade_count = 0
daily_trade_date  = None

# ── Win rate feedback — weight adjustment ─────────────────────

def update_signal_weights():
    """
    After enough trades, adjust signal weights based on win rate.
    Signals with >60% win rate get boosted. <40% get reduced.
    Only adjusts after 10+ trades to avoid noise.
    """
    global signal_weights
    total = trade_stats["total"]
    if total < 10:
        return  # not enough data yet

    new_weights = {}
    for name, perf in signal_performance.items():
        sig_total = perf["wins"] + perf["losses"]
        if sig_total < 5:
            new_weights[name] = BASE_SIGNAL_WEIGHTS[name]
            continue
        win_rate = perf["wins"] / sig_total
        base     = BASE_SIGNAL_WEIGHTS[name]
        if   win_rate > 0.65: adj = min(base * 1.3, base + 0.5)   # boost up to 30%
        elif win_rate > 0.55: adj = min(base * 1.1, base + 0.2)   # small boost
        elif win_rate < 0.35: adj = max(base * 0.7, base - 0.4)   # reduce up to 30%
        elif win_rate < 0.45: adj = max(base * 0.9, base - 0.2)   # small reduce
        else:                 adj = base                            # neutral
        new_weights[name] = round(adj, 2)

    if new_weights != signal_weights:
        print(f"Signal weights updated: {new_weights}")
        signal_weights = new_weights

def record_trade_signals(ticker, was_win, active_signals):
    """Record which signals were active on this trade for feedback."""
    for sig_name in active_signals:
        if sig_name in signal_performance:
            if was_win:
                signal_performance[sig_name]["wins"] += 1
            else:
                signal_performance[sig_name]["losses"] += 1
    update_signal_weights()

# ── Account helpers ───────────────────────────────────────────

def get_account():
    try:
        return api.get_account()
    except Exception as e:
        print(f"get_account error: {e}")
        return None

def get_account_size():
    acct = get_account()
    return min(float(acct.equity), MAX_ACCOUNT) if acct else MAX_ACCOUNT

def get_available_cash():
    acct = get_account()
    if not acct: return 0
    return min(float(acct.cash), float(acct.buying_power))

def get_account_state():
    try:
        acct = get_account()
        if not acct:
            return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime}
        return {
            "portfolio": round(float(acct.equity), 2),
            "cash":      round(float(acct.cash), 2),
            "pnl":       round(float(acct.equity) - float(acct.last_equity), 2),
            "regime":    market_regime,
        }
    except Exception as e:
        print(f"get_account_state error: {e}")
        return {"portfolio":0,"cash":0,"pnl":0,"regime":market_regime}

# ── Dynamic scaling ───────────────────────────────────────────

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

# ── Window + market helpers ───────────────────────────────────

def in_trading_window():
    now = datetime.now(NY)
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return (TRADING_START_H*60+TRADING_START_M) <= mins < (TRADING_END_H*60+TRADING_END_M)

def is_market_open():
    try:
        return api.get_clock().is_open
    except Exception as e:
        print(f"is_market_open error: {e}")
        return False

def is_pre_market():
    """Returns True between 8:00–9:30 AM ET on weekdays."""
    now  = datetime.now(NY)
    if now.weekday() >= 5: return False
    mins = now.hour * 60 + now.minute
    return (8*60) <= mins < (9*60+30)

# ── Cooldown helpers ──────────────────────────────────────────

def set_cooldown(ticker, reason="signal"):
    d = {"stop_loss":COOLDOWN_STOP,"take_profit":COOLDOWN_PROFIT,
         "signal":COOLDOWN_SIGNAL,"eod_close":COOLDOWN_SIGNAL}
    cooldowns[ticker] = {"until":time.time()+d.get(reason,COOLDOWN_SIGNAL),"reason":reason}

def is_on_cooldown(ticker):
    cd = cooldowns.get(ticker)
    return bool(cd and time.time() < cd["until"])

def cooldown_remaining(ticker):
    cd = cooldowns.get(ticker)
    if not cd: return 0
    return max(0, int(cd["until"] - time.time()))

# ── Pre-market gap scanner ────────────────────────────────────

def run_gap_scan():
    """
    Runs at 8:45 AM before market open.
    Finds stocks gapping 2-15% with elevated relative volume.
    These are the highest-probability intraday momentum setups.
    """
    global gap_candidates, active_tickers
    print("=== Pre-market gap scan starting ===")
    acct_size = get_account_size()
    floor     = get_price_floor(acct_size)
    ceiling   = get_price_ceiling(acct_size)
    candidates = {}

    # Check our known universe for gaps
    from scanner import SEED_UNIVERSE
    check_list = list(set(SEED_UNIVERSE + list(active_tickers)))

    for ticker in check_list:
        try:
            # Get yesterday's close
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=5)
            bars  = api.get_bars(ticker, "1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=5, feed="iex").df

            if bars is None or bars.empty or len(bars) < 2:
                continue
            if hasattr(bars.index, 'levels'):
                if ticker in bars.index.get_level_values(0):
                    bars = bars.loc[ticker]
                else:
                    continue

            prev_close = float(bars.iloc[-1]["close"])
            avg_vol    = float(bars["volume"].mean())

            # Get pre-market / latest bar
            latest = api.get_latest_bar(ticker, feed="iex")
            cur_price = float(latest.c)
            cur_vol   = float(latest.v)

            if not (floor <= cur_price <= ceiling):
                continue

            gap_pct  = (cur_price - prev_close) / prev_close * 100
            rvol     = cur_vol / (avg_vol / 390) if avg_vol > 0 else 1  # vs avg min volume

            # Must gap within our range AND have volume
            if GAP_MIN_PCT <= abs(gap_pct) <= GAP_MAX_PCT and rvol >= GAP_MIN_RVOL:
                candidates[ticker] = {
                    "ticker":     ticker,
                    "price":      round(cur_price, 2),
                    "prev_close": round(prev_close, 2),
                    "gap_pct":    round(gap_pct, 2),
                    "rvol":       round(rvol, 2),
                    "direction":  "up" if gap_pct > 0 else "down",
                }
                print(f"  GAP: {ticker} ${prev_close}→${cur_price} "
                      f"({gap_pct:+.1f}%) rvol={rvol:.1f}x")
            time.sleep(0.2)
        except Exception as e:
            continue

    gap_candidates = candidates
    print(f"=== Gap scan complete: {len(candidates)} candidates ===")

    # Merge gap candidates into active tickers
    if candidates:
        # Prioritize gap stocks — add to front of active list
        gap_tickers    = [t for t in candidates if candidates[t]["direction"] == "up"][:3]
        current        = [t for t in active_tickers if t not in gap_tickers]
        active_tickers = (gap_tickers + current)[:5]
        for t in gap_tickers:
            price_history.setdefault(t,  [])
            volume_history.setdefault(t, [])
        print(f"Active tickers with gaps: {active_tickers}")

    socketio.emit("gaps", {
        "candidates": list(candidates.values()),
        "scanned_at": datetime.now(NY).strftime("%I:%M %p ET"),
    })

# ── Real-time relative volume ─────────────────────────────────

def update_rvol():
    """
    Every 15 minutes during market hours, check if active tickers
    are trading above their average volume pace.
    A stock trading 2x normal volume is "in play" — better signals.
    """
    global rvol_cache
    print("Updating relative volume...")
    for ticker in list(active_tickers):
        try:
            end   = datetime.now(pytz.utc) - timedelta(minutes=20)
            start = end - timedelta(days=10)
            bars  = api.get_bars(ticker, "1Day",
                        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        limit=10, feed="iex").df
            if bars is None or bars.empty: continue
            if hasattr(bars.index, 'levels'):
                if ticker in bars.index.get_level_values(0):
                    bars = bars.loc[ticker]
                else: continue

            avg_vol   = float(bars["volume"].mean())
            latest    = api.get_latest_bar(ticker, feed="iex")
            today_vol = float(latest.v)

            # Adjust for time of day — normalize to full day equivalent
            now       = datetime.now(NY)
            mins_open = max(1, (now.hour - 9) * 60 + now.minute - 30)
            projected = today_vol * (390 / mins_open) if mins_open < 390 else today_vol
            rvol      = projected / avg_vol if avg_vol > 0 else 1.0
            rvol_cache[ticker] = round(rvol, 2)
            print(f"  RVOL {ticker}: {rvol:.2f}x (proj={int(projected):,} avg={int(avg_vol):,})")
            time.sleep(0.2)
        except Exception as e:
            print(f"  RVOL error {ticker}: {e}")

# ── Filters ───────────────────────────────────────────────────

def passes_filters(ticker, price, account_size=None):
    if account_size is None: account_size = get_account_size()
    floor   = get_price_floor(account_size)
    ceiling = get_price_ceiling(account_size)
    min_vol = get_min_volume(account_size)
    if not (floor <= price <= ceiling):
        print(f"  {ticker} filtered — ${price:.2f} outside ${floor}–${ceiling:.2f}")
        return False
    grade = ticker_grades.get(ticker)
    if grade and grade not in MIN_GRADE:
        print(f"  {ticker} filtered — grade {grade}")
        return False
    # Relative volume check — skip stocks with no activity
    rvol = rvol_cache.get(ticker)
    if rvol is not None and rvol < RVOL_THRESHOLD:
        print(f"  {ticker} filtered — low rvol {rvol:.2f}x")
        return False
    try:
        end   = datetime.now(pytz.utc) - timedelta(minutes=20)
        start = end - timedelta(days=5)
        bars  = api.get_bars(ticker, "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=5, feed="iex").df
        if bars is None or bars.empty: return False
        if hasattr(bars.index, 'levels'):
            if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
            else: return False
        if float(bars["volume"].mean()) < min_vol:
            print(f"  {ticker} low avg volume")
            return False
        return True
    except Exception as e:
        print(f"  Filter error {ticker}: {e}")
        return False

# ── Position sizing ───────────────────────────────────────────

def position_size(price, account_size=None, pred_score=0, rvol=1.0):
    """
    Sizes based on prediction AND relative volume.
    High rvol = stock is in play = larger position allowed.
    """
    try:
        if account_size is None: account_size = get_account_size()
        usable = min(get_available_cash(), MAX_ACCOUNT)

        if   pred_score >= PRED_STRONG_BUY: trade_pct = 0.65
        elif pred_score >= PRED_NORMAL_BUY: trade_pct = MAX_TRADE_PCT
        elif pred_score >= PRED_REDUCE:     trade_pct = 0.40
        else:                               trade_pct = 0.30

        # Boost size if stock is running hot (high rvol)
        if rvol >= 3.0:   trade_pct = min(trade_pct * 1.2, 0.70)
        elif rvol >= 2.0: trade_pct = min(trade_pct * 1.1, 0.65)

        # Spread penalty for very cheap stocks
        if   price < 1.00: trade_pct *= 0.80
        elif price < 2.00: trade_pct *= 0.90

        max_per_trade = usable * trade_pct
        if max_per_trade < price:
            print(f"  Can't afford ${price:.2f} (max: ${max_per_trade:.2f})")
            return 0
        return max(0, int(max_per_trade / price))
    except Exception as e:
        print(f"  position_size error: {e}")
        return 0

# ── Indicators ────────────────────────────────────────────────

def calc_rsi(prices, period=14):
    if len(prices) < period+1: return 50
    gains=losses=0
    for i in range(-period, 0):
        d=prices[i]-prices[i-1]
        if d>0: gains+=d
        else:   losses+=abs(d)
    if losses==0: return 100
    return 100-(100/(1+gains/losses))

def calc_ma(prices, n):
    s=prices[-n:] if len(prices)>=n else prices
    return sum(s)/len(s)

def calc_bollinger(prices, n=20):
    s=prices[-n:] if len(prices)>=n else prices
    mean=sum(s)/len(s)
    std=(sum((v-mean)**2 for v in s)/len(s))**0.5
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
    hist=price_history.get(ticker,[])
    vols=volume_history.get(ticker,[])
    hist_len=len(hist)
    if hist_len < 5:
        return [{"name":n,"action":"hold","signal":50} for n in signal_weights]

    rsi              = calc_rsi(hist)
    ma50             = calc_ma(hist, min(50, hist_len))
    ma200            = calc_ma(hist, min(200, hist_len))
    mean,upper,lower = calc_bollinger(hist)
    vwap             = calc_vwap(hist[-78:], vols[-78:])
    macd_line        = calc_macd(hist)
    z_score          = (price-mean)/max((upper-mean),0.01)
    ok_buy           = market_regime in ("trending_up","ranging")
    ok_sell          = market_regime in ("trending_down","ranging")
    ma_conf          = min(1.0, hist_len/20)

    def act(bc, sc):
        if bc and ok_buy:  return "buy"
        if sc and ok_sell: return "sell"
        return "hold"

    return [
        {"name":"MA Crossover",
         "action":act(ma50>ma200*1.005, ma200>ma50*1.005) if ma_conf>0.5 else "hold",
         "signal":min(95,50+(ma50-ma200)/max(ma200,1)*600)},
        {"name":"RSI",
         "action":act(rsi<35, rsi>65),
         "signal":100-rsi},
        {"name":"Bollinger",
         "action":act(price<lower*0.99, price>upper*1.01),
         "signal":max(5,min(95,50-z_score*40))},
        {"name":"VWAP",
         "action":act(price>vwap*1.001, price<vwap*0.999),
         "signal":min(95,max(5,50+(price-vwap)/max(vwap,1)*500))},
        {"name":"MACD",
         "action":act(macd_line>0, macd_line<0),
         "signal":min(95,max(5,50+macd_line*10))},
        {"name":"Mean Reversion",
         "action":act(price<mean*0.96, price>mean*1.04),
         "signal":min(92,max(8,50+(mean-price)/max(mean,1)*250))},
    ]

# ── Weighted voting ───────────────────────────────────────────

def weighted_vote(signals):
    buy_w=sell_w=hold_w=0.0
    for sig in signals:
        w=signal_weights.get(sig.get("name",""),1.0)
        if   sig.get("action")=="buy":  buy_w  += w
        elif sig.get("action")=="sell": sell_w += w
        else:                           hold_w += w
    return round(buy_w,2), round(sell_w,2), round(hold_w,2)

# ── Market regime ─────────────────────────────────────────────

def update_market_regime():
    global market_regime
    REGIME_TICKERS=["SIRI","TELL","NIO","MARA","SOFI","AMC","BB","NOK"]
    try:
        end=datetime.now(pytz.utc)-timedelta(minutes=20)
        start=end-timedelta(days=15)
        ss=start.strftime("%Y-%m-%dT%H:%M:%SZ")
        es=end.strftime("%Y-%m-%dT%H:%M:%SZ")
        closes=None
        for ticker in REGIME_TICKERS:
            try:
                bars=api.get_bars(ticker,"1Day",start=ss,end=es,limit=12,feed="iex").df
                if bars is None or bars.empty: continue
                if hasattr(bars.index,'levels'):
                    if ticker in bars.index.get_level_values(0): bars=bars.loc[ticker]
                    else: continue
                if len(bars)>=5:
                    closes=list(bars["close"])
                    print(f"Regime using {ticker}")
                    break
            except: continue
        if closes is None:
            all_p=[]
            for t in active_tickers:
                h=price_history.get(t,[])
                if len(h)>=5: all_p.extend(h[-10:])
            closes=all_p if len(all_p)>=5 else None
        if closes is None:
            print("Regime: no data — keeping:", market_regime); return
        ma5=calc_ma(closes,min(5,len(closes)))
        ma10=calc_ma(closes,min(10,len(closes)))
        latest=closes[-1]
        if   ma5>ma10*1.005 and latest>ma5: market_regime="trending_up"
        elif ma5<ma10*0.995 and latest<ma5: market_regime="trending_down"
        else:                               market_regime="ranging"
        print(f"Regime: {market_regime}")
    except Exception as e:
        print(f"Regime error: {e}")

def update_market_regime_with_retry(attempts=5, delay=8):
    global market_regime
    for i in range(attempts):
        update_market_regime()
        if market_regime!="unknown": return
        time.sleep(delay)

# ── Stops ─────────────────────────────────────────────────────

def check_stops(ticker, price):
    pos=open_positions.get(ticker)
    if not pos: return
    # Trailing stop — move stop up as price rises
    if price > pos["entry"] * 1.05:   # once up 5%, trail stop to breakeven+1%
        new_stop = max(pos["stop"], pos["entry"] * 1.01)
        if new_stop > pos["stop"]:
            open_positions[ticker]["stop"] = round(new_stop, 3)
            print(f"  Trailing stop moved: {ticker} stop → ${new_stop:.3f}")
    if price <= pos["stop"]:
        print(f"STOP-LOSS: {ticker} @ ${price:.2f}")
        force_sell(ticker, price, reason="stop_loss")
    elif price >= pos["target"]:
        print(f"TAKE-PROFIT: {ticker} @ ${price:.2f}")
        force_sell(ticker, price, reason="take_profit")

def force_sell(ticker, price, reason="stop_loss"):
    try:
        pos=api.get_position(ticker)
        qty=int(pos.qty)
        if qty<=0:
            open_positions.pop(ticker,None); return
        entry=float(pos.avg_entry_price)
        pnl=round((price-entry)*qty,2)
        api.submit_order(symbol=ticker,qty=qty,side="sell",
                         type="market",time_in_force="day")

        # Record which signals were active for feedback
        was_win = pnl > 0
        pos_data = open_positions.get(ticker, {})
        active_sigs = pos_data.get("active_signals", [])
        record_trade_signals(ticker, was_win, active_sigs)

        trade_log.insert(0,{
            "type":"SELL","ticker":ticker,"qty":qty,
            "price":round(price,2),"pnl":pnl,"reason":reason,
            "ts":int(time.time()*1000)
        })
        open_positions.pop(ticker,None)
        set_cooldown(ticker,reason)

        trade_stats["total"]    += 1
        trade_stats["total_pnl"] = round(trade_stats["total_pnl"]+pnl, 2)
        if pnl>0:
            trade_stats["wins"]+=1
            trade_stats["best_trade"]=max(trade_stats["best_trade"],pnl)
        else:
            trade_stats["losses"]+=1
            trade_stats["worst_trade"]=min(trade_stats["worst_trade"],pnl)

        wr=trade_stats["wins"]/max(trade_stats["total"],1)*100
        print(f"SELL {qty}x {ticker} @ ${price:.2f} | {reason} | PnL ${pnl:+.2f} | WR:{wr:.0f}%")
    except Exception as e:
        print(f"Force sell error {ticker}: {e}")

def close_all_positions_eod():
    try:
        for pos in api.list_positions():
            force_sell(pos.symbol,float(pos.current_price),reason="eod_close")
        print("EOD: all positions closed")
    except Exception as e:
        print(f"EOD close error: {e}")

# ── Decision engine ───────────────────────────────────────────

def make_decision(ticker, signals, price):
    buy_w,sell_w,_ = weighted_vote(signals)
    pred    = prediction_cache.get(ticker,{})
    pscore  = pred.get("score",0)
    tf_bias = pred.get("tf_bias",0)
    rvol    = rvol_cache.get(ticker,1.0)
    is_gap  = ticker in gap_candidates

    # Skip if prediction strongly bearish
    if pscore <= PRED_SKIP:
        return "hold", f"pred_skip({pscore})", buy_w, sell_w

    # Dynamic threshold based on prediction + gap + rvol
    if pscore >= 40:
        buy_threshold = sell_threshold = 1.2
    elif pscore >= 20:
        buy_threshold = sell_threshold = 1.5
    elif pscore < PRED_NEED_CONF:
        buy_threshold  = 2.5
        sell_threshold = 2.0
    else:
        buy_threshold  = BASE_BUY_THRESHOLD
        sell_threshold = BASE_SELL_THRESHOLD

    # Gap stock bonus — lower threshold since momentum is confirmed
    if is_gap and gap_candidates[ticker]["direction"]=="up":
        buy_threshold  = max(1.0, buy_threshold - 0.4)
        print(f"  {ticker} gap bonus — buy threshold → {buy_threshold:.1f}")

    # High rvol bonus — stock is active, signals are more reliable
    if rvol >= 2.0:
        buy_threshold  = max(1.0, buy_threshold - 0.2)
        sell_threshold = max(1.0, sell_threshold - 0.2)

    # Daily downtrend raises buy bar
    if tf_bias==-1 and buy_w>=buy_threshold:
        buy_threshold += 1.0

    if   buy_w  >= buy_threshold:  action="buy"
    elif sell_w >= sell_threshold: action="sell"
    else:                          action="hold"

    reason=(f"buy_w={buy_w:.1f}>={buy_threshold:.1f}"  if action=="buy"
            else f"sell_w={sell_w:.1f}>={sell_threshold:.1f}" if action=="sell"
            else f"hold(b={buy_w:.1f},s={sell_w:.1f})")
    return action, reason, buy_w, sell_w

# ── Trade execution ───────────────────────────────────────────

def execute(ticker, action, price, signals, reason="signal"):
    try:
        if action=="buy" and ticker not in open_positions:
            if is_on_cooldown(ticker):
                print(f"  {ticker} cooldown {cooldown_remaining(ticker)}s"); return
            acct_size=get_account_size()
            if not passes_filters(ticker,price,acct_size): return
            pred_score=prediction_cache.get(ticker,{}).get("score",0)
            rvol=rvol_cache.get(ticker,1.0)
            qty=position_size(price,acct_size,pred_score,rvol)
            if qty==0:
                print(f"  Skipping {ticker} — qty 0"); return
            stop,target,atr=calculate_stops(api,ticker,price,
                                            STOP_LOSS_PCT,TAKE_PROFIT_PCT)
            api.submit_order(symbol=ticker,qty=qty,side="buy",
                             type="market",time_in_force="day")
            # Store which signals were active for feedback later
            active_sigs=[s["name"] for s in signals if s["action"]=="buy"]
            open_positions[ticker]={
                "entry":price,"stop":stop,"target":target,"qty":qty,
                "atr":atr,"active_signals":active_sigs
            }
            rr=round((target-price)/(price-stop),2) if price>stop else "?"
            is_gap=ticker in gap_candidates
            trade_log.insert(0,{
                "type":"BUY","ticker":ticker,"qty":qty,
                "price":round(price,2),"pnl":None,
                "stop":stop,"target":target,
                "pred_score":pred_score,"rvol":rvol,
                "gap":is_gap,"reason":reason,
                "ts":int(time.time()*1000)
            })
            print(f"BUY {qty}x {ticker} @ ${price:.2f} | SL${stop} TP${target} "
                  f"| R:R={rr} | pred={pred_score:+.0f} | rvol={rvol:.1f}x"
                  +(" | GAP PLAY" if is_gap else ""))
        elif action=="sell" and ticker in open_positions:
            force_sell(ticker,price,reason=reason or "signal")
    except Exception as e:
        print(f"Order error {ticker}: {e}")

# ── Apply scan + validate fallbacks ──────────────────────────

def apply_scan_results(results_today, acct_size=None):
    global active_tickers
    if acct_size is None: acct_size=get_account_size()
    ceiling=get_price_ceiling(acct_size)
    floor=get_price_floor(acct_size)
    affordable=[s for s in results_today
                if floor<=s["price"]<=ceiling
                and s.get("grade","F") in MIN_GRADE]
    if affordable:
        new_tickers=[s["ticker"] for s in affordable[:5]]
        for s in affordable[:5]:
            ticker_grades[s["ticker"]]=s.get("grade","C")
        for t in new_tickers:
            price_history.setdefault(t,[])
            volume_history.setdefault(t,[])
        active_tickers=new_tickers
        print(f"Active tickers: {active_tickers}")
    else:
        print("No affordable scan results — keeping current tickers")

def validate_fallback_tickers():
    global active_tickers
    acct_size=get_account_size()
    floor=get_price_floor(acct_size)
    ceiling=get_price_ceiling(acct_size)
    valid=[]
    for ticker in FALLBACK_TICKERS:
        try:
            bar=api.get_latest_bar(ticker,feed="iex")
            price=float(bar.c)
            if floor<=price<=ceiling:
                valid.append(ticker)
                print(f"  Fallback {ticker} OK @ ${price:.2f}")
            else:
                print(f"  Fallback {ticker} out of range @ ${price:.2f}")
        except Exception as e:
            print(f"  Fallback check error {ticker}: {e}")
    active_tickers=valid if valid else ["SIRI","TELL","AMC","BB","NOK"]
    print(f"Active tickers set to: {active_tickers}")

# ── Pre-market + RVOL loop ────────────────────────────────────

def premarket_loop():
    """
    Handles pre-market gap scan (8:45 AM) and
    intraday relative volume updates (every 15 min).
    """
    gap_scanned_today   = None
    rvol_updated_hours  = set()
    last_rvol_day       = None

    while True:
        try:
            now = datetime.now(NY)
            day = now.date()

            # Reset daily trackers
            if day != last_rvol_day:
                rvol_updated_hours.clear()
                last_rvol_day = day

            # Pre-market gap scan at 8:45 AM
            if (now.weekday() < 5
                    and now.hour == 8 and now.minute >= 45
                    and gap_scanned_today != day):
                gap_scanned_today = day
                try:
                    run_gap_scan()
                except Exception as e:
                    print(f"Gap scan error: {e}")

            # Relative volume update every 15 minutes during trading window
            if (now.weekday() < 5
                    and in_trading_window()
                    and is_market_open()):
                bucket = now.hour * 4 + now.minute // 15  # unique 15-min bucket
                if bucket not in rvol_updated_hours:
                    rvol_updated_hours.add(bucket)
                    try:
                        update_rvol()
                        socketio.emit("rvol", rvol_cache)
                    except Exception as e:
                        print(f"RVOL update error: {e}")

        except Exception as e:
            print(f"Premarket loop error: {e}")
        time.sleep(60)

# ── Prediction loop ───────────────────────────────────────────

def prediction_loop():
    global prediction_cache
    predicted_hours=set(); last_pred_day=None; last_tickers=[]
    while True:
        try:
            now=datetime.now(NY); hour=now.hour; day=now.date()
            if day!=last_pred_day:
                predicted_hours.clear(); last_pred_day=day
            tickers_changed=set(active_tickers)!=set(last_tickers)
            should_run=(now.weekday()<5 and in_trading_window() and
                        ((hour in PRED_HOURS and hour not in predicted_hours)
                         or tickers_changed))
            if should_run:
                predicted_hours.add(hour)
                last_tickers=list(active_tickers)
                tickers=list(active_tickers)
                print(f"Running predictions for {tickers}...")
                try:
                    results=run_predictions(api,tickers,market_regime)
                    prediction_cache.update(results)
                    summary=[(t,f"{r.get('score',0):+.0f}") for t,r in results.items()]
                    print(f"Predictions: {summary}")
                    socketio.emit("predictions",{
                        t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                           "confidence":r.get("confidence","low"),
                           "components":r.get("components",{}),"signals":r.get("signals",[]),
                           "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail",""),
                           "timestamp":r.get("timestamp","")}
                        for t,r in results.items()
                    })
                except Exception as e:
                    print(f"Prediction run error: {e}")
        except Exception as e:
            print(f"Prediction loop error: {e}")
        time.sleep(60)

# ── Scanner loop ──────────────────────────────────────────────

def scanner_loop():
    global scan_results,active_tickers,market_regime
    scanned_hours=set(); last_scan_day=None
    print("Scanner loop started")
    while True:
        try:
            now=datetime.now(NY); hour=now.hour; day=now.date()
            if day!=last_scan_day:
                scanned_hours.clear(); last_scan_day=day
            if (now.weekday()<5 and hour in SCAN_HOURS
                    and hour not in scanned_hours and in_trading_window()):
                scanned_hours.add(hour)
                print(f"Scanner firing at {now.strftime('%H:%M')} ET...")
                try:
                    update_market_regime()
                    acct_size=get_account_size()
                    scan_results=run_full_scan(api)
                    apply_scan_results(scan_results.get("today",[]),acct_size)
                    socketio.emit("scan",scan_results)
                    print(f"Scan complete | regime:{market_regime}")
                except Exception as e:
                    print(f"Scanner error: {e}")
        except Exception as e:
            print(f"Scanner loop error: {e}")
        time.sleep(60)

# ── On connect ────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("Client connected")
    try:
        market_open=is_market_open()
        window_open=in_trading_window()
        both_open=market_open and window_open
        print(f"on_connect: market={market_open} window={window_open} tickers={active_tickers}")
        state={
            "tickers":{},"account":get_account_state(),
            "trades":trade_log[:40],
            "market_status":"open" if both_open else "closed",
        }
        if both_open:
            for ticker in list(active_tickers):
                try:
                    bar=api.get_latest_bar(ticker,feed="iex")
                    price=float(bar.c)
                    price_history.setdefault(ticker,[]).append(price)
                    volume_history.setdefault(ticker,[]).append(float(bar.v))
                    sigs=get_signals(ticker,price)
                    action,reason,buy_w,sell_w=make_decision(ticker,sigs,price)
                    pos=open_positions.get(ticker)
                    pred=prediction_cache.get(ticker,{})
                    state["tickers"][ticker]={
                        "price":round(price,2),"signals":sigs,
                        "action":action,"buy_weight":buy_w,"sell_weight":sell_w,
                        "stop":pos["stop"] if pos else None,
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
        socketio.emit("state",state)
        if scan_results.get("scanned_at"): socketio.emit("scan",scan_results)
        if prediction_cache:
            socketio.emit("predictions",{
                t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                   "confidence":r.get("confidence","low"),
                   "components":r.get("components",{}),"signals":r.get("signals",[]),
                   "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail","")}
                for t,r in prediction_cache.items()
            })
        if gap_candidates:
            socketio.emit("gaps",{
                "candidates":list(gap_candidates.values()),
                "scanned_at":datetime.now(NY).strftime("%I:%M %p ET"),
            })
        if rvol_cache: socketio.emit("rvol",rvol_cache)
    except Exception as e:
        print(f"on_connect error: {e}")

# ── Main bot loop ─────────────────────────────────────────────

def bot_loop():
    global market_regime
    last_regime_update=None
    while True:
        try:
            now=datetime.now(NY)
            market_open=is_market_open()
            window_open=in_trading_window()
            both_open=market_open and window_open

            print(f"Loop: {now.strftime('%H:%M')} ET | market={market_open} window={window_open} | tickers={active_tickers} | regime={market_regime}")

            if not window_open:
                socketio.emit("state",{
                    "tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Trading window closed — resumes 8:45 AM ET"
                })
                time.sleep(60); continue

            if not market_open:
                socketio.emit("state",{
                    "tickers":{},"account":get_account_state(),
                    "trades":trade_log[:40],"market_status":"closed",
                    "message":"Warming up — market opens 9:30 AM ET"
                })
                time.sleep(30); continue

            if (not last_regime_update or
                    (now-last_regime_update).total_seconds()>1800):
                update_market_regime()
                last_regime_update=now

            if now.hour==13 and now.minute>=50:
                close_all_positions_eod()
                print("EOD close done.")
                time.sleep(600); continue

            state={"tickers":{},"account":{},"market_status":"open","regime":market_regime}

            for ticker in list(active_tickers):
                try:
                    bar=api.get_latest_bar(ticker,feed="iex")
                    price=float(bar.c); vol=float(bar.v)
                    price_history.setdefault(ticker,[]).append(price)
                    volume_history.setdefault(ticker,[]).append(vol)
                    if len(price_history[ticker])>200: price_history[ticker].pop(0)
                    if len(volume_history[ticker])>200: volume_history[ticker].pop(0)

                    check_stops(ticker,price)
                    sigs=get_signals(ticker,price)
                    action,reason,buy_w,sell_w=make_decision(ticker,sigs,price)

                    if action!="hold":
                        execute(ticker,action,price,sigs,reason=reason)

                    pos=open_positions.get(ticker)
                    pred=prediction_cache.get(ticker,{})
                    cd=cooldown_remaining(ticker)
                    rvol=rvol_cache.get(ticker,None)
                    n_buys=sum(1 for s in sigs if s["action"]=="buy")
                    n_sells=sum(1 for s in sigs if s["action"]=="sell")

                    state["tickers"][ticker]={
                        "price":round(price,2),"signals":sigs,
                        "action":action,"buy_weight":buy_w,"sell_weight":sell_w,
                        "stop":pos["stop"] if pos else None,
                        "target":pos["target"] if pos else None,
                        "cooldown":cd,
                        "grade":ticker_grades.get(ticker,"—"),
                        "pred_score":pred.get("score",None),
                        "pred_label":pred.get("label","—"),
                        "pred_conf":pred.get("confidence","—"),
                        "tf_bias":pred.get("tf_bias",0),
                        "rvol":rvol,
                        "is_gap":ticker in gap_candidates,
                    }
                    print(f"  {ticker}: ${price:.2f} | {action} | "
                          f"votes={n_buys}b/{n_sells}s w={buy_w:.1f}b/{sell_w:.1f}s | "
                          f"{market_regime} | pred={pred.get('score',0):+.0f}"
                          +(f" | rvol={rvol:.1f}x" if rvol else "")
                          +(f" | GAP" if ticker in gap_candidates else "")
                          +(f" | cd:{cd}s" if cd else ""))
                except Exception as e:
                    print(f"  {ticker} error: {e}")

            state["account"]=get_account_state()
            state["trades"]=trade_log[:40]
            socketio.emit("state",state)

        except Exception as e:
            print(f"Loop error: {e}")
        time.sleep(INTERVAL)

# ── Flask routes ──────────────────────────────────────────────

@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/scanner")
def scanner_page(): return render_template("scanner.html")

@app.route("/predictions")
def predictions_page(): return render_template("predictions.html")

@app.route("/state")
def state_json(): return jsonify({"trades":trade_log[:20]})

@app.route("/scan")
def scan_json(): return jsonify(scan_results)

@app.route("/predictions/data")
def predictions_json(): return jsonify(prediction_cache)

@app.route("/gaps")
def gaps_json(): return jsonify({
    "candidates":list(gap_candidates.values()),
    "scanned_at":datetime.now(NY).strftime("%I:%M %p ET") if gap_candidates else None
})

@app.route("/stats")
def stats_json():
    total=max(trade_stats["total"],1)
    return jsonify({
        **trade_stats,
        "win_rate": round(trade_stats["wins"]/total*100,1),
        "signal_weights": signal_weights,
        "signal_performance": signal_performance,
    })

@app.route("/scan/manual",methods=["POST"])
def manual_scan():
    def run():
        global scan_results,active_tickers,market_regime
        print("Manual scan started...")
        try:
            update_market_regime()
            acct_size=get_account_size()
            new_scan=run_full_scan(api)
            def merge(existing,incoming):
                seen={s["ticker"] for s in incoming}
                kept=[s for s in existing if s["ticker"] not in seen]
                return (incoming+kept)[:10]
            scan_results["today"]        =merge(scan_results.get("today",[]),new_scan["today"])
            scan_results["yesterday"]    =merge(scan_results.get("yesterday",[]),new_scan["yesterday"])
            scan_results["scanned_at"]   =new_scan["scanned_at"]
            scan_results["account_size"] =new_scan.get("account_size",acct_size)
            scan_results["price_range"]  =new_scan.get("price_range","—")
            scan_results["universe_size"]=new_scan.get("universe_size",0)
            scan_results["manual"]       =True
            apply_scan_results(scan_results["today"],acct_size)
            socketio.emit("scan",scan_results)
            print(f"Manual scan complete | {len(scan_results['today'])} picks")
        except Exception as e:
            print(f"Manual scan error: {e}")
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"}),202

@app.route("/predictions/manual",methods=["POST"])
def manual_predictions():
    def run():
        tickers=list(active_tickers)
        try:
            results=run_predictions(api,tickers,market_regime)
            prediction_cache.update(results)
            summary=[(t,f"{r.get('score',0):+.0f}") for t,r in results.items()]
            print(f"Manual predictions: {summary}")
            socketio.emit("predictions",{
                t:{"score":r.get("score",0),"label":r.get("label","neutral"),
                   "confidence":r.get("confidence","low"),
                   "components":r.get("components",{}),"signals":r.get("signals",[]),
                   "tf_bias":r.get("tf_bias",0),"tf_detail":r.get("tf_detail","")}
                for t,r in results.items()
            })
        except Exception as e:
            print(f"Manual predictions error: {e}")
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started"}),202

@app.route("/ping")
def ping(): return "pong",200

# ── Startup ───────────────────────────────────────────────────

if __name__ == "__main__":
    port=int(os.environ.get("PORT",10000))
    print(f"=== BOT STARTING v6 | port {port} ===")
    print(f"=== FALLBACK_TICKERS: {FALLBACK_TICKERS} ===")

    def startup():
        time.sleep(3)
        print("=== STARTUP: regime detection ===")
        update_market_regime()
        print("=== STARTUP: validating tickers ===")
        validate_fallback_tickers()
        print(f"=== STARTUP COMPLETE | tickers={active_tickers} | regime={market_regime} ===")

    threading.Thread(target=startup,         daemon=True).start()
    threading.Thread(target=bot_loop,        daemon=True).start()
    threading.Thread(target=scanner_loop,    daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=premarket_loop,  daemon=True).start()

    socketio.run(app,host="0.0.0.0",port=port)