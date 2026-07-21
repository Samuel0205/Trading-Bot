"""
replay.py — offline expectancy harness for the LIVE strategy.

Replays historical 1-minute IEX bars through the bot's REAL decision pipeline
(get_signals → make_decision, imported from bot.py — never copied, so this can
never drift stale like the old backtest.py fork did) and simulates the live
execution mechanics:

  - 16-bar signal warmup, 3-tick decision debounce
  - limit entry at price*1.003 (filled within 3 bars or abandoned)
  - the 2:1 bracket from predictions.calculate_stops (real function, fed by a
    lookahead-safe daily-bar cache built from the same minute data)
  - partial exit of half at +1R → stop to breakeven, banked PnL
  - trailing stop 3% below high-water after +5%
  - stop checked BEFORE target when both hit inside one bar (conservative)
  - min-hold 300s for signal sells, EOD flat at 15:28 ET
  - slippage penalty on all market exits (default 0.1%)

Run ON THE SERVER (needs Alpaca keys in the environment):

    BOT_REPLAY=1 python3 replay.py --days 30 --tickers SOFI,MARA,RIVN,PLTR,ENPH
    BOT_REPLAY=1 python3 replay.py --days 30 --tickers AUTO   # names from trades DB

HONEST LIMITATIONS
  - Predictions cannot be replayed historically (news/FinBERT are point-in-time),
    so the harness SWEEPS a fixed prediction score across --pred-sweep levels and
    reports expectancy per level. This bounds the answer instead of faking it.
  - One decision per 1-min bar approximates the live 30s cadence.
  - market_regime is held at "ranging" (the conservative threshold, ×1.2).
  - IEX minute bars cover ~2-3% of consolidated volume — same feed the live bot
    trades on, so gating behaves comparably.
"""
import os
os.environ["BOT_REPLAY"] = "1"

import argparse, sys, time
from datetime import datetime, timedelta

import pandas as pd
import pytz

import bot                      # real signal + decision code
import predictions              # real calculate_stops

NY = pytz.timezone("America/New_York")

SLIPPAGE_PCT   = 0.001          # applied against us on every market exit
RISK_PER_TRADE = bot.RISK_PER_TRADE
PERSIST        = bot.ACTION_PERSIST_TICKS
WARMUP         = bot.SIGNAL_WARMUP_BARS


def fetch_min_bars(api, ticker, days):
    """1-min IEX bars for the last `days` trading days, cached to CSV."""
    cache = f".replay_cache_{ticker}_{days}d.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df if not df.empty else None
    end   = datetime.now(pytz.utc) - timedelta(minutes=20)
    start = end - timedelta(days=days + 8)
    try:
        df = api.get_bars(ticker, "1Min",
                          start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          feed="iex", limit=100_000).df
        if df is None or df.empty:
            return None
        if hasattr(df.index, "levels"):
            if ticker in df.index.get_level_values(0):
                df = df.loc[ticker]
            else:
                return None
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"  fetch {ticker}: {e}")
        return None


def build_daily(df_min):
    """Aggregate minute bars → daily OHLCV keyed by ET date (for ATR/stops)."""
    et = df_min.index.tz_convert(NY)
    g  = df_min.groupby(et.date)
    return pd.DataFrame({
        "open":   g["open"].first(),  "high": g["high"].max(),
        "low":    g["low"].min(),     "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })


def make_stops(daily, sim_date, entry_price):
    """calculate_stops' clamp fed only with bars BEFORE sim_date (no lookahead)."""
    hist = daily[daily.index < sim_date].tail(10)
    atr  = None
    if len(hist) >= 5:
        a = predictions.calc_atr(hist, period=min(10, len(hist) - 1))
        if a and a > 0:
            atr = a
    raw   = (atr * 1.5) if atr else (entry_price * 0.05)
    dist  = min(max(raw, entry_price * 0.015), entry_price * 0.04)
    return round(entry_price - dist, 3), round(entry_price + dist * 2, 3)


def replay_ticker(ticker, df_min, daily, pred_score):
    """Run one ticker through the real pipeline. Returns list of closed trades."""
    trades  = []
    pos     = None       # {entry, stop, target, qty, half, banked, high, since}
    streak  = {"action": None, "count": 0}
    bot.prediction_cache[ticker] = {"score": pred_score, "confidence": "medium",
                                    "tf_bias": 0, "fetched_at": time.time()}
    bot.market_regime = "ranging"
    bot.gap_candidates.pop(ticker, None)

    et_index = df_min.index.tz_convert(NY)
    avg_day_vol = daily["volume"].mean() if len(daily) else 0

    for sim_date, day_df in df_min.groupby(et_index.date):
        bot.price_history[ticker]  = []
        bot.volume_history[ticker] = []
        streak = {"action": None, "count": 0}
        day_et = day_df.index.tz_convert(NY)
        cum_vol = 0.0
        bars = list(day_df.itertuples())

        for i, bar in enumerate(bars):
            t = day_et[i]
            if (t.hour, t.minute) < (9, 35) or (t.hour, t.minute) >= (15, 28):
                continue
            price = float(bar.close)
            cum_vol += float(bar.volume)
            bot.price_history[ticker].append(price)
            bot.volume_history[ticker].append(float(bar.volume))
            if len(bot.price_history[ticker]) > 200:
                bot.price_history[ticker].pop(0)
                bot.volume_history[ticker].pop(0)

            mins_open = max(1, (t.hour - 9) * 60 + t.minute - 30)
            if avg_day_vol > 0:
                bot.rvol_cache[ticker] = round(cum_vol * (390 / mins_open) / avg_day_vol, 2)

            # ── manage open position (mirrors check_stops) ──
            if pos:
                hi, lo = float(bar.high), float(bar.low)
                pos["high"] = max(pos["high"], hi)
                one_r = pos["entry"] - pos["stop_orig"]
                if not pos["half_done"] and hi >= pos["entry"] + one_r:
                    fill = pos["entry"] + one_r
                    pos["banked"] += (fill - pos["entry"]) * pos["half"]
                    pos["qty"]  -= pos["half"]
                    pos["stop"]  = round(pos["entry"] * 1.005, 3)
                    pos["half_done"] = True
                if (pos["high"] - pos["entry"]) / pos["entry"] > 0.05:
                    pos["stop"] = max(pos["stop"], round(pos["high"] * 0.97, 3))
                exit_px, why = None, None
                if lo <= pos["stop"]:            # stop before target: conservative
                    exit_px, why = pos["stop"] * (1 - SLIPPAGE_PCT), "stop"
                elif hi >= pos["target"]:
                    exit_px, why = pos["target"] * (1 - SLIPPAGE_PCT), "target"
                elif (t.hour, t.minute) >= (15, 27):
                    exit_px, why = price * (1 - SLIPPAGE_PCT), "eod"
                if exit_px:
                    pnl = (exit_px - pos["entry"]) * pos["qty"] + pos["banked"]
                    trades.append({"ticker": ticker, "pnl": round(pnl, 2),
                                   "why": why, "date": str(sim_date)})
                    pos = None
                    continue

            # ── decisions through the REAL pipeline ──
            if len(bot.price_history[ticker]) < WARMUP:
                continue
            sigs = bot.get_signals(ticker, price)
            action, _r, _b, _s = bot.make_decision(ticker, sigs, price)
            if action == "hold":
                streak = {"action": None, "count": 0}
                continue
            if streak["action"] == action:
                streak["count"] += 1
            else:
                streak = {"action": action, "count": 1}
            if streak["count"] < PERSIST:
                continue

            if action == "buy" and pos is None:
                limit = price * 1.003
                for j in range(i + 1, min(i + 4, len(bars))):
                    nb = bars[j]
                    if float(nb.low) <= limit:
                        entry = min(float(nb.open), limit)
                        stop, target = make_stops(daily, sim_date, entry)
                        qty = int(RISK_PER_TRADE / max(entry - stop, 0.01))
                        if qty < 1:
                            break
                        pos = {"entry": entry, "stop": stop, "stop_orig": stop,
                               "target": target, "qty": qty, "half": qty // 2 or 1,
                               "banked": 0.0, "high": entry, "half_done": False,
                               "since": i}
                        break
            elif action == "sell" and pos is not None and (i - pos["since"]) >= 10:
                exit_px = price * (1 - SLIPPAGE_PCT)
                pnl = (exit_px - pos["entry"]) * pos["qty"] + pos["banked"]
                trades.append({"ticker": ticker, "pnl": round(pnl, 2),
                               "why": "signal", "date": str(sim_date)})
                pos = None
    return trades


def stats(trades):
    if not trades:
        return "  no trades"
    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    aw = sum(wins) / len(wins) if wins else 0
    al = sum(losses) / len(losses) if losses else 0
    exits = {}
    for t in trades:
        exits[t["why"]] = exits.get(t["why"], 0) + 1
    return (f"  trades={len(pnls)}  WR={len(wins)/len(pnls)*100:.0f}%  "
            f"avg_win=${aw:+.2f}  avg_loss=${al:+.2f}  "
            f"expectancy=${sum(pnls)/len(pnls):+.2f}/trade  "
            f"total=${sum(pnls):+.2f}  exits={exits}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--tickers", default="AUTO")
    ap.add_argument("--pred-sweep", default="10,20,40")
    args = ap.parse_args()

    api = bot.api
    if args.tickers == "AUTO":
        try:
            from database import get_recent_trades, init_db
            init_db()
            tickers = sorted({t["ticker"] for t in get_recent_trades(days=45)})[:12]
        except Exception:
            tickers = []
        if not tickers:
            tickers = ["SOFI", "MARA", "RIVN", "PLTR", "ENPH", "HOOD"]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    sweep = [int(s) for s in args.pred_sweep.split(",")]
    print(f"Replaying {args.days}d × {tickers} × pred∈{sweep}\n")

    data = {}
    for t in tickers:
        df = fetch_min_bars(api, t, args.days)
        if df is not None and len(df) > 500:
            data[t] = (df, build_daily(df))
            print(f"  {t}: {len(df)} minute bars")
        else:
            print(f"  {t}: insufficient data — skipped")

    for S in sweep:
        all_trades = []
        for t, (df, daily) in data.items():
            all_trades += replay_ticker(t, df, daily, S)
        print(f"\npred_score={S}:")
        print(stats(all_trades))
        by = {}
        for tr in all_trades:
            by.setdefault(tr["ticker"], []).append(tr["pnl"])
        for t, pnls in sorted(by.items(), key=lambda kv: sum(kv[1])):
            print(f"    {t:6s} n={len(pnls):3d}  ${sum(pnls):+9.2f}")


if __name__ == "__main__":
    main()
