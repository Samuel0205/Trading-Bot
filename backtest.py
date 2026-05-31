"""
backtest.py — Strategy backtester

Replays 3 months of daily bar data bar-by-bar through the exact
same signal logic, weighted voting, ATR stops, trailing stops,
and cooldowns used by the live bot.

Limitations (honest):
  - Daily bars only (minute bars need paid Alpaca tier)
  - Sentiment approximated via keyword scoring
  - 0.5% friction applied per trade to simulate spread
  - Prediction scores set to neutral (0) since historical
    FinBERT is too slow to run for 3 months of data

FIX vs original:
  - Equity curve merge was mathematically wrong (added val - allocation
    repeatedly, double-counting deltas). Now uses return-weighted merge:
    each ticker's equity curve is converted to a return multiplier,
    and the combined curve reflects the weighted portfolio return.

Output: full trade log + stats for each ticker universe,
        plus a comparison between them.
"""

import time
from datetime import datetime, timedelta
import pytz

# ── Config ────────────────────────────────────────────────────

STARTING_CAPITAL  = 20.00
MAX_TRADE_PCT     = 0.50
STOP_LOSS_PCT     = 0.05
TAKE_PROFIT_PCT   = 0.10
FRICTION_PCT      = 0.005
COOLDOWN_BARS     = 3
MIN_PROFIT_PCT    = 0.03
THRESHOLD         = 2.0   # matches BASE_BUY_THRESHOLD in bot.py
MAX_TRADES_PER_TICKER = 30

MONTHS_BACK = 3

SIGNAL_WEIGHTS = {
    "MA Crossover":   0.8,
    "RSI":            1.2,
    "Bollinger":      1.0,
    "VWAP":           1.5,
    "MACD":           1.0,
    "Mean Reversion": 0.8,
}

CURRENT_TICKERS = ["AMC","BB","NOK","OCGN","SHLS","SIRI","TELL","MVIS","NIO","CLOV"]

BROAD_UNIVERSE = [
    "SIRI","TELL","AMC","BB","NOK","MVIS","NIO","MARA","SOFI","CLOV",
    "NKLA","OCGN","SHLS","IDEX","GNUS","NAKD","KOSS","XELA","EXPR",
    "PLTR","RIVN","LCID","RIOT","ACB","CGC","TLRY","SNDL","SPCE","CHPT"
]

# ── Indicators (identical to live bot) ───────────────────────

def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    gains = losses = 0
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
    s    = prices[-n:] if len(prices) >= n else prices
    mean = sum(s) / len(s)
    std  = (sum((v - mean)**2 for v in s) / len(s)) ** 0.5
    return mean, mean + 2*std, mean - 2*std

def calc_vwap(prices, volumes):
    if not prices or not volumes or sum(volumes) == 0:
        return prices[-1] if prices else 0
    return sum(p*v for p,v in zip(prices,volumes)) / sum(volumes)

def calc_macd(prices):
    def ema(data, period):
        if len(data) < period: return data[-1] if data else 0
        k = 2 / (period + 1)
        val = sum(data[:period]) / period
        for p in data[period:]: val = p*k + val*(1-k)
        return val
    if len(prices) < 26: return 0
    return ema(prices, 12) - ema(prices, 26)

def calc_atr(highs, lows, closes, period=10):
    if len(closes) < 2: return closes[-1] * 0.02 if closes else 0
    trs = []
    for i in range(1, min(len(closes), period+1)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i-1]),
            abs(lows[i]   - closes[i-1])
        ))
    return sum(trs) / len(trs) if trs else closes[-1] * 0.02

RSI_OVERBOUGHT_VETO  = 75
RSI_OVERSOLD_VETO    = 25
PRICE_EXTENSION_VETO = 0.08

def get_signals(hist_closes, hist_volumes, price):
    if len(hist_closes) < 5:
        return {n: "hold" for n in SIGNAL_WEIGHTS}

    rsi              = calc_rsi(hist_closes)
    ma50             = calc_ma(hist_closes, min(50, len(hist_closes)))
    ma200            = calc_ma(hist_closes, min(200, len(hist_closes)))
    mean, upper, lower = calc_bollinger(hist_closes)
    vwap             = calc_vwap(hist_closes[-20:], hist_volumes[-20:])
    macd_line        = calc_macd(hist_closes)
    ma_conf          = min(1.0, len(hist_closes) / 20)

    # Veto conditions — mirrors live bot logic
    rsi_overbought = rsi > RSI_OVERBOUGHT_VETO
    rsi_oversold   = rsi < RSI_OVERSOLD_VETO
    price_extended = (price - mean) / max(mean, 0.01) > PRICE_EXTENSION_VETO

    def act(bc, sc, veto_buy=False, veto_sell=False):
        if bc and not veto_buy:  return "buy"
        if sc and not veto_sell: return "sell"
        return "hold"

    signals = {
        "MA Crossover":   act(ma50>ma200*1.005, ma200>ma50*1.005,
                              veto_buy=rsi_overbought or price_extended) if ma_conf > 0.5 else "hold",
        "RSI":            act(rsi<35, rsi>65),
        "Bollinger":      act(price<lower*0.99, price>upper*1.01,
                              veto_buy=rsi_overbought or price_extended,
                              veto_sell=rsi_oversold),
        "VWAP":           act(price>vwap*1.001, price<vwap*0.999,
                              veto_buy=rsi_overbought or price_extended),
        "MACD":           act(macd_line>0, macd_line<0,
                              veto_buy=rsi_overbought or price_extended),
        "Mean Reversion": act(price<mean*0.96, price>mean*1.04,
                              veto_buy=rsi_overbought, veto_sell=rsi_oversold),
    }
    # Global veto: RSI overbought + price extended → all buys off
    if rsi_overbought and price_extended:
        signals = {k: "hold" if v == "buy" else v for k, v in signals.items()}
    return signals

def weighted_vote(signal_actions):
    buy_w = sell_w = 0.0
    for name, action in signal_actions.items():
        w = SIGNAL_WEIGHTS.get(name, 1.0)
        if   action == "buy":  buy_w  += w
        elif action == "sell": sell_w += w
    return round(buy_w, 2), round(sell_w, 2)

# ── Data fetcher ──────────────────────────────────────────────

def fetch_bars(api, ticker):
    try:
        end   = datetime.now(pytz.utc) - timedelta(days=1)
        start = end - timedelta(days=MONTHS_BACK * 31)
        bars  = api.get_bars(
            ticker, "1Day",
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            feed="iex"
        ).df

        if bars is None or bars.empty:
            return None

        if hasattr(bars.index, 'levels'):
            if ticker in bars.index.get_level_values(0):
                bars = bars.loc[ticker]
            else:
                return None

        return bars if len(bars) >= 30 else None
    except Exception as e:
        print(f"  fetch_bars error {ticker}: {e}")
        return None

# ── Single ticker backtest ────────────────────────────────────

def backtest_ticker(ticker, bars, starting_cash):
    closes  = list(bars["close"])
    highs   = list(bars["high"])
    lows    = list(bars["low"])
    volumes = list(bars["volume"])
    dates   = list(bars.index)

    cash         = starting_cash
    position     = None
    trades       = []
    cooldown_bar = -1
    trade_count  = 0
    equity_curve = [starting_cash]

    hist_closes  = []
    hist_volumes = []

    for i, (close, high, low, vol, date) in enumerate(
            zip(closes, highs, lows, volumes, dates)):

        hist_closes.append(close)
        hist_volumes.append(vol)

        pos_value = position["qty"] * close if position else 0
        equity_curve.append(round(cash + pos_value, 4))

        if position:
            # Trailing stop
            if close > position["entry"] * 1.05:
                new_stop = max(position["stop"], position["entry"] * 1.01)
                position["stop"] = new_stop

            # Stop loss
            if close <= position["stop"]:
                proceeds = position["qty"] * close * (1 - FRICTION_PCT)
                pnl      = round(proceeds - position["cost"], 4)
                cash    += proceeds
                trades.append({
                    "ticker": ticker, "entry": position["entry"], "exit": close,
                    "qty": position["qty"], "pnl": pnl, "reason": "stop_loss",
                    "bars_held": i - position["entry_idx"],
                    "date_exit": str(date)[:10], "date_entry": position["date_entry"],
                })
                position     = None
                cooldown_bar = i + COOLDOWN_BARS
                continue

            # Take profit
            if close >= position["target"]:
                proceeds = position["qty"] * close * (1 - FRICTION_PCT)
                pnl      = round(proceeds - position["cost"], 4)
                cash    += proceeds
                trades.append({
                    "ticker": ticker, "entry": position["entry"], "exit": close,
                    "qty": position["qty"], "pnl": pnl, "reason": "take_profit",
                    "bars_held": i - position["entry_idx"],
                    "date_exit": str(date)[:10], "date_entry": position["date_entry"],
                })
                position     = None
                cooldown_bar = i + COOLDOWN_BARS
                continue

        if i <= cooldown_bar or trade_count >= MAX_TRADES_PER_TICKER:
            continue
        if len(hist_closes) < 10:
            continue

        sigs          = get_signals(hist_closes[:-1], hist_volumes[:-1], close)
        buy_w, sell_w = weighted_vote(sigs)

        if not position and buy_w >= THRESHOLD and cash >= close:
            atr    = calc_atr(highs[max(0,i-10):i+1], lows[max(0,i-10):i+1], closes[max(0,i-10):i+1])
            stop   = close - (atr * 1.5) if atr > 0 else close * (1 - STOP_LOSS_PCT)
            target = close + (atr * 3.0) if atr > 0 else close * (1 + TAKE_PROFIT_PCT)
            stop   = max(stop,   close * 0.90)
            stop   = min(stop,   close * 0.98)
            target = max(target, close * (1 + MIN_PROFIT_PCT))

            if (target - close) / close < MIN_PROFIT_PCT:
                continue

            max_spend = cash * MAX_TRADE_PCT
            qty       = max(0, int(max_spend / close))
            if qty == 0: continue

            cost  = qty * close * (1 + FRICTION_PCT)
            if cost > cash: continue

            cash -= cost
            position = {
                "entry":      close, "stop": stop, "target": target,
                "qty":        qty,   "cost": cost, "entry_idx": i,
                "date_entry": str(date)[:10],
            }
            trade_count += 1

        elif position and sell_w >= THRESHOLD:
            proceeds = position["qty"] * close * (1 - FRICTION_PCT)
            pnl      = round(proceeds - position["cost"], 4)
            cash    += proceeds
            trades.append({
                "ticker": ticker, "entry": position["entry"], "exit": close,
                "qty": position["qty"], "pnl": pnl, "reason": "signal",
                "bars_held": i - position["entry_idx"],
                "date_exit": str(date)[:10], "date_entry": position["date_entry"],
            })
            position     = None
            cooldown_bar = i + COOLDOWN_BARS

    # Force close at end
    if position and closes:
        last_close = closes[-1]
        proceeds   = position["qty"] * last_close * (1 - FRICTION_PCT)
        pnl        = round(proceeds - position["cost"], 4)
        cash      += proceeds
        trades.append({
            "ticker": ticker, "entry": position["entry"], "exit": last_close,
            "qty": position["qty"], "pnl": pnl, "reason": "end_of_test",
            "bars_held": len(closes) - position["entry_idx"],
            "date_exit": str(dates[-1])[:10], "date_entry": position["date_entry"],
        })

    return trades, round(cash, 4), equity_curve

# ── Stats calculator ──────────────────────────────────────────

def calc_stats(trades, starting_capital, final_equity, equity_curve):
    if not trades:
        return {
            "total_trades":0,"win_rate":0,"total_return":0,"return_pct":0,
            "max_drawdown":0,"max_dd_pct":0,"sharpe":0,"avg_hold_bars":0,
            "best_trade":0,"worst_trade":0,"wins":0,"losses":0,"avg_win":0,"avg_loss":0,
        }

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    pnls      = [t["pnl"] for t in trades]
    total_ret = final_equity - starting_capital

    peak   = starting_capital
    max_dd = 0.0
    for val in equity_curve:
        if val > peak: peak = val
        dd = peak - val
        if dd > max_dd: max_dd = dd

    if len(equity_curve) > 1:
        daily_rets = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                      for i in range(1, len(equity_curve))
                      if equity_curve[i-1] > 0]
        if daily_rets:
            avg_ret = sum(daily_rets) / len(daily_rets)
            std_ret = (sum((r - avg_ret)**2 for r in daily_rets) / len(daily_rets)) ** 0.5
            sharpe  = round((avg_ret / std_ret) * (252**0.5), 2) if std_ret > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    avg_hold = round(sum(t["bars_held"] for t in trades) / len(trades), 1)

    return {
        "total_trades":  len(trades),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "total_return":  round(total_ret, 2),
        "return_pct":    round(total_ret / starting_capital * 100, 1),
        "max_drawdown":  round(max_dd, 2),
        "max_dd_pct":    round(max_dd / starting_capital * 100, 1),
        "sharpe":        sharpe,
        "avg_hold_bars": avg_hold,
        "best_trade":    round(max(pnls), 2),
        "worst_trade":   round(min(pnls), 2),
        "wins":          len(wins),
        "losses":        len(losses),
        "avg_win":       round(sum(t["pnl"] for t in wins)   / max(len(wins),   1), 2),
        "avg_loss":      round(sum(t["pnl"] for t in losses) / max(len(losses), 1), 2),
    }

# ── Master backtest runner ────────────────────────────────────

def run_backtest(api, universe_name="current"):
    universes = {}
    if universe_name in ("current", "both"):
        universes["Current tickers"] = CURRENT_TICKERS
    if universe_name in ("broad", "both"):
        universes["Broad universe"] = BROAD_UNIVERSE

    results = {}

    for name, tickers in universes.items():
        print(f"\n=== Backtesting {name} ({len(tickers)} tickers) ===")
        all_trades     = []
        tickers_tested = []

        # Per-ticker allocation
        allocation = STARTING_CAPITAL / len(tickers)

        # Store each ticker's equity curve as return multipliers
        # FIX: original merged curves by adding (val - allocation) which double-counted
        # New approach: convert each curve to returns, sum weighted returns into portfolio curve
        all_curves = []  # list of (allocation, equity_curve)

        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            bars = fetch_bars(api, ticker)
            if bars is None:
                print(f"  {ticker}: no data — skipping")
                continue

            t_trades, t_final, t_curve = backtest_ticker(ticker, bars, allocation)

            all_trades.extend(t_trades)
            all_curves.append((allocation, t_curve))
            tickers_tested.append({
                "ticker":   ticker,
                "trades":   len(t_trades),
                "pnl":      round(t_final - allocation, 2),
                "final":    round(t_final, 2),
                "win_rate": round(
                    len([t for t in t_trades if t["pnl"]>0]) / max(len(t_trades), 1) * 100, 1
                ),
            })
            time.sleep(0.3)

        # FIX: build combined equity curve correctly
        # Convert each ticker's curve to fractional returns from its start
        # Then reconstruct portfolio curve: sum of (allocation * return_multiplier)
        if all_curves:
            max_len = max(len(c) for _, c in all_curves)
            portfolio_curve = []
            for i in range(max_len):
                port_val = 0
                for alloc, curve in all_curves:
                    if i < len(curve):
                        # Return relative to start of this ticker's allocation
                        ret_mult  = curve[i] / curve[0] if curve[0] > 0 else 1.0
                        port_val += alloc * ret_mult
                    else:
                        # Use final value if curve ended early
                        ret_mult  = curve[-1] / curve[0] if curve[0] > 0 else 1.0
                        port_val += alloc * ret_mult
                portfolio_curve.append(round(port_val, 2))
        else:
            portfolio_curve = [STARTING_CAPITAL]

        all_trades.sort(key=lambda t: t.get("date_entry",""), reverse=True)
        final_equity = portfolio_curve[-1] if portfolio_curve else STARTING_CAPITAL
        stats        = calc_stats(all_trades, STARTING_CAPITAL, final_equity, portfolio_curve)

        results[name] = {
            "stats":          stats,
            "trades":         all_trades[:50],
            "equity_curve":   [round(v, 2) for v in portfolio_curve],
            "tickers_tested": sorted(tickers_tested, key=lambda t: t["pnl"], reverse=True),
        }
        print(f"=== {name} done | {stats['total_trades']} trades | "
              f"return {stats['return_pct']:+.1f}% | win rate {stats['win_rate']}% ===")

    return {
        "results":  results,
        "config": {
            "starting_capital": STARTING_CAPITAL,
            "months_back":      MONTHS_BACK,
            "threshold":        THRESHOLD,
            "stop_loss_pct":    STOP_LOSS_PCT,
            "take_profit_pct":  TAKE_PROFIT_PCT,
            "friction_pct":     FRICTION_PCT,
            "cooldown_bars":    COOLDOWN_BARS,
        },
        "run_at": datetime.now(pytz.timezone("America/New_York")).strftime("%I:%M %p ET %b %d"),
    }
