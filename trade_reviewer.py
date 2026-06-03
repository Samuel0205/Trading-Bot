"""
trade_reviewer.py — Live trade analysis and improvement suggestions

Analyzes actual trade history stored in the database to answer:
  - Which tickers are consistently losing? Which are profitable?
  - Which hours of day produce the best/worst results?
  - Are we getting whipsawed? (same ticker, multiple same-day entries)
  - Do we exit winners too early? (avg win vs avg loss ratio)
  - Which exit reasons dominate? (stop vs take_profit vs signal)
  - What's the hold time for wins vs losses?

Returns structured analysis + prioritized improvement suggestions.
Cached for REVIEW_TTL seconds.
"""

import time
from datetime import datetime
from collections import defaultdict
import pytz

REVIEW_TTL = 1800   # 30 min

NY = pytz.timezone("America/New_York")

_CACHE    = {}
_CACHE_TS = 0.0


def _et_hour(ts_ms):
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=NY)
        return dt.hour
    except Exception:
        return -1


def _hold_hours(trade):
    e = trade.get("entry_ts")
    x = trade.get("exit_ts")
    if e and x and x > e:
        return (x - e) / 3_600_000
    return None


def analyze_trades(trades):
    """
    Accepts a list of trade dicts from database.get_recent_trades().
    Returns:
      {summary, ticker_stats, hour_stats, reason_stats,
       hold_stats, chop_events, suggestions, analyzed_at}
    """
    closed = [t for t in trades if t.get("pnl") is not None]

    if not closed:
        return {
            "summary": {}, "ticker_stats": {}, "hour_stats": {},
            "reason_stats": {}, "hold_stats": {}, "chop_events": [],
            "suggestions": ["No closed trades yet — let the bot run for a day and come back."],
            "analyzed_at": datetime.now(NY).strftime("%I:%M %p ET"),
        }

    # ── Summary ───────────────────────────────────────────────
    pnls   = [t["pnl"] for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    avg_win  = sum(wins)   / len(wins)   if wins   else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    rr       = abs(avg_win / avg_loss)   if avg_loss != 0 else 0

    hold_win  = [h for t in closed for h in [_hold_hours(t)] if h and t["pnl"] > 0]
    hold_loss = [h for t in closed for h in [_hold_hours(t)] if h and t["pnl"] <= 0]

    summary = {
        "total":      len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   round(len(wins) / len(closed) * 100, 1),
        "total_pnl":  round(sum(pnls), 2),
        "avg_win":    round(avg_win,  2),
        "avg_loss":   round(avg_loss, 2),
        "rr_ratio":   round(rr, 2),
        "avg_hold_win_h":  round(sum(hold_win)  / len(hold_win),  1) if hold_win  else None,
        "avg_hold_loss_h": round(sum(hold_loss) / len(hold_loss), 1) if hold_loss else None,
    }

    # ── Per-ticker ─────────────────────────────────────────────
    by_ticker = defaultdict(list)
    for t in closed:
        by_ticker[t.get("ticker", "?")].append(t["pnl"])

    ticker_stats = {}
    for sym, tp in by_ticker.items():
        w = [p for p in tp if p > 0]
        l = [p for p in tp if p <= 0]
        ticker_stats[sym] = {
            "trades":    len(tp),
            "win_rate":  round(len(w) / len(tp) * 100, 1),
            "total_pnl": round(sum(tp), 2),
            "avg_pnl":   round(sum(tp) / len(tp), 2),
            "wins":      len(w),
            "losses":    len(l),
        }

    # ── Per-hour-of-day ────────────────────────────────────────
    by_hour = defaultdict(list)
    for t in closed:
        h = _et_hour(t["entry_ts"]) if t.get("entry_ts") else -1
        if h >= 0:
            by_hour[h].append(t["pnl"])

    hour_stats = {}
    for h in sorted(by_hour.keys()):
        tp = by_hour[h]
        w  = [p for p in tp if p > 0]
        hour_stats[str(h)] = {
            "label":     f"{h % 12 or 12}{'am' if h < 12 else 'pm'}",
            "trades":    len(tp),
            "win_rate":  round(len(w) / len(tp) * 100, 1) if tp else 0,
            "total_pnl": round(sum(tp), 2),
        }

    # ── Exit reasons ───────────────────────────────────────────
    by_reason = defaultdict(list)
    for t in closed:
        by_reason[t.get("reason", "unknown")].append(t["pnl"])

    reason_stats = {}
    for reason, tp in by_reason.items():
        w = [p for p in tp if p > 0]
        reason_stats[reason] = {
            "count":     len(tp),
            "win_rate":  round(len(w) / len(tp) * 100, 1) if tp else 0,
            "total_pnl": round(sum(tp), 2),
            "avg_pnl":   round(sum(tp) / len(tp), 2),
        }

    # ── Hold time stats ────────────────────────────────────────
    hold_stats = {
        "win_avg_h":  round(sum(hold_win)  / len(hold_win),  2) if hold_win  else None,
        "loss_avg_h": round(sum(hold_loss) / len(hold_loss), 2) if hold_loss else None,
        "win_samples":  len(hold_win),
        "loss_samples": len(hold_loss),
    }

    # ── Chop detection ─────────────────────────────────────────
    # Flag days where the same ticker was entered 3+ times (whipsaw pattern)
    day_ticker_entries = defaultdict(lambda: defaultdict(int))
    for t in closed:
        day = t.get("date") or (
            datetime.fromtimestamp(t["entry_ts"] / 1000).strftime("%Y-%m-%d")
            if t.get("entry_ts") else None
        )
        if day:
            day_ticker_entries[day][t.get("ticker", "?")] += 1

    chop_events = []
    for day, tickers in day_ticker_entries.items():
        for sym, cnt in tickers.items():
            if cnt >= 3:
                chop_events.append({"date": day, "ticker": sym, "entries": cnt})
    chop_events.sort(key=lambda x: x["date"], reverse=True)

    # ── Suggestions ────────────────────────────────────────────
    suggestions = []

    wr = summary["win_rate"]
    if wr < 38 and len(closed) >= 5:
        suggestions.append({
            "priority": "HIGH",
            "area": "Win Rate",
            "finding": f"Win rate is {wr:.0f}% — below breakeven for a 1:1 R:R strategy.",
            "action": "Raise BASE_BUY_THRESHOLD to 2.5 in bot.py so the bot requires "
                      "stronger signal consensus before entering.",
        })
    elif wr > 62 and len(closed) >= 8:
        suggestions.append({
            "priority": "LOW",
            "area": "Win Rate",
            "finding": f"Win rate is strong at {wr:.0f}%.",
            "action": "Consider increasing TAKE_PROFIT_PCT from 12% to 15–18% to capture "
                      "more of each winning move.",
        })

    rr_r = summary["rr_ratio"]
    if rr_r < 0.8 and len(wins) >= 3 and len(losses) >= 3:
        suggestions.append({
            "priority": "HIGH",
            "area": "Risk:Reward",
            "finding": f"Avg win (${avg_win:.2f}) < avg loss (${avg_loss:.2f}). R:R = {rr_r:.2f}.",
            "action": "The bot exits winners too early. Try increasing PARTIAL_EXIT_PCT "
                      "from 6% to 9%, or disabling partial exits on gap plays.",
        })

    # Underperforming tickers
    bad = sorted(
        [(sym, d) for sym, d in ticker_stats.items()
         if d["trades"] >= 3 and d["win_rate"] < 35 and d["total_pnl"] < -1.0],
        key=lambda x: x[1]["total_pnl"]
    )
    if bad:
        names = ", ".join(f"{sym} ({d['win_rate']:.0f}% WR, ${d['total_pnl']:.2f})"
                          for sym, d in bad[:3])
        suggestions.append({
            "priority": "MEDIUM",
            "area": "Stock Selection",
            "finding": f"Consistently losing tickers: {names}.",
            "action": "Add these to FALLBACK_TICKERS blacklist or raise their individual "
                      "entry threshold via the prediction skip gate (PRED_SKIP).",
        })

    # Good tickers
    good = sorted(
        [(sym, d) for sym, d in ticker_stats.items()
         if d["trades"] >= 3 and d["win_rate"] >= 60 and d["total_pnl"] > 1.0],
        key=lambda x: -x[1]["win_rate"]
    )
    if good:
        names = ", ".join(f"{sym} ({d['win_rate']:.0f}% WR)" for sym, d in good[:3])
        suggestions.append({
            "priority": "LOW",
            "area": "Stock Selection",
            "finding": f"Best performers: {names}.",
            "action": "These tickers are working. Make sure they're included in the "
                      "seed universe and are not accidentally filtered out.",
        })

    # Stop loss dominance — if >60% of exits are stop losses
    stop_data  = reason_stats.get("stop_loss", {})
    tp_data    = reason_stats.get("take_profit", {})
    total_exit = len(closed)
    if stop_data.get("count", 0) / max(total_exit, 1) > 0.60:
        suggestions.append({
            "priority": "MEDIUM",
            "area": "Stop Placement",
            "finding": f"{stop_data['count']} of {total_exit} trades ({stop_data['count']/total_exit*100:.0f}%) "
                       "exit via stop loss — entries are too early or stop too tight.",
            "action": "Try widening STOP_LOSS_PCT from 5% to 6–7%, or require VWAP "
                      "confirmation before entering (already gated for gap plays).",
        })

    # Chop pattern
    if chop_events:
        total_extra = sum(c["entries"] - 2 for c in chop_events)
        suggestions.append({
            "priority": "HIGH",
            "area": "Chop / Whipsaw",
            "finding": f"Detected {len(chop_events)} day(s) with 3+ entries in the same ticker "
                       f"(~{total_extra} avoidable re-entries).",
            "action": "The anti-chop gate (2 stops = blocked for the day) is already "
                      "active. Check that CHOP_BLOCK_THRESHOLD = 2 in bot.py and that "
                      "cooldown after stop loss is at least 15 minutes.",
        })

    # Bad hours
    bad_hours = [(h, d) for h, d in hour_stats.items()
                 if d["trades"] >= 3 and d["win_rate"] < 35]
    if bad_hours:
        labels = ", ".join(d["label"] for _, d in
                           sorted(bad_hours, key=lambda x: x[1]["win_rate"])[:2])
        suggestions.append({
            "priority": "MEDIUM",
            "area": "Time of Day",
            "finding": f"Weak trading hours: {labels} ET (win rate <35%).",
            "action": "Extend NO_NEW_ENTRY_MINS or add those hours to a time-based "
                      "blackout. Consider only trading 10am–2pm ET.",
        })

    # Hold time divergence
    if (hold_stats["win_avg_h"] and hold_stats["loss_avg_h"] and
            hold_stats["win_avg_h"] < hold_stats["loss_avg_h"]):
        suggestions.append({
            "priority": "LOW",
            "area": "Hold Time",
            "finding": f"Wins held {hold_stats['win_avg_h']}h avg, losses held "
                       f"{hold_stats['loss_avg_h']}h avg. Losses are held longer than wins.",
            "action": "Let winners run: raise TAKE_PROFIT_PCT. Cut losers faster: "
                      "tighten STOP_LOSS_PCT or add a time-stop (close if flat after 2h).",
        })

    # Nothing wrong
    if not suggestions:
        suggestions.append({
            "priority": "LOW",
            "area": "Overall",
            "finding": "No critical issues detected in recent trades.",
            "action": "Keep monitoring. Check signal win rates in the /stats endpoint "
                      "to see which individual signals are performing best.",
        })

    return {
        "summary":      summary,
        "ticker_stats": ticker_stats,
        "hour_stats":   hour_stats,
        "reason_stats": reason_stats,
        "hold_stats":   hold_stats,
        "chop_events":  chop_events[:10],
        "suggestions":  suggestions,
        "analyzed_at":  datetime.now(NY).strftime("%I:%M %p ET"),
        "trade_count":  len(closed),
    }


def get_trade_review(trades):
    """
    Run analysis on the given trades list. Caches result for REVIEW_TTL seconds.
    Always refreshes if trades list has grown.
    """
    global _CACHE, _CACHE_TS
    now     = time.time()
    n_new   = len([t for t in trades if t.get("pnl") is not None])
    n_cached = _CACHE.get("trade_count", -1)
    if now - _CACHE_TS < REVIEW_TTL and n_new == n_cached and _CACHE:
        return _CACHE
    result    = analyze_trades(trades)
    _CACHE    = result
    _CACHE_TS = now
    return result
