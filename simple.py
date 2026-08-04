"""
simple.py — the whole strategy, in one page.

ONE RULE:
    Own the 2 strongest big ETFs that are trading above their 200-day average.
    Anything that falls below its 200-day average gets sold. Otherwise: cash.

That's it. No signals, no votes, no predictions, no sentiment analysis, no gap
scanner, no rvol, no correlation matrix, no debounce, no cooldowns, no circuit
breakers, no partial exits, no end-of-day flush.

WHY THIS REPLACED THE OLD STRATEGY
The intraday version lost money for a structural reason no amount of tuning
fixes: ~500 round trips per month through the spread on thin small caps costs
more than any edge available at that timeframe, and going flat every afternoon
forfeits the overnight drift where much of the long-run equity return has
historically accrued. This version trades a handful of times per YEAR, on
penny-spread ETFs, and holds overnight. Friction stops being the story.

WHAT IT DOES AND DOESN'T DO
The 200-day trend filter is not a way to beat buy-and-hold in a bull market —
in a straight-up market it WILL lag, because it re-enters late after a dip and
occasionally sits in cash during a whipsaw. What it has historically done is
cut the depth of bear-market drawdowns, because it mechanically exits and stays
out while a downtrend persists. If the market rips upward for a year, holding
VUG/QQQ yourself will very likely beat this. That's the honest trade-off:
this is a discipline machine, not an alpha machine.

Signals are computed on COMPLETED daily bars only (today's partial bar is
excluded), and the check runs once per trading day. There is nothing to babysit.
"""

import os
from datetime import datetime, timedelta

import pytz

NY = pytz.timezone("America/New_York")

# The only knobs. Defaults are deliberately boring.
UNIVERSE      = [t.strip().upper() for t in
                 os.environ.get("SIMPLE_UNIVERSE", "QQQ,VUG,SPY,VTI,IWM").split(",")]
MAX_HOLDINGS  = int(os.environ.get("SIMPLE_MAX_HOLDINGS", "2"))
TREND_DAYS    = 200     # long-term trend filter
MOMENTUM_DAYS = 126     # ~6 months, used only to rank which uptrends to own
INVEST_PCT    = 0.98    # keep a small cash buffer so orders never bounce


def _daily_closes(api, ticker, need=TREND_DAYS + 40):
    """Completed daily closes, oldest→newest. Today's partial bar is excluded."""
    end   = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(need * 1.6) + 30)
    bars  = api.get_bars(
        ticker, "1Day",
        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        feed="iex", limit=need + 50,
    ).df
    if bars is None or bars.empty:
        return None
    if hasattr(bars.index, "levels"):
        if ticker in bars.index.get_level_values(0):
            bars = bars.loc[ticker]
        else:
            return None
    closes = [float(c) for c in bars["close"]]
    return closes if len(closes) >= TREND_DAYS + 1 else None


def evaluate(api, ticker):
    """Everything we need to know about one ETF."""
    closes = _daily_closes(api, ticker)
    if not closes:
        return None
    price  = closes[-1]
    sma    = sum(closes[-TREND_DAYS:]) / TREND_DAYS
    past   = closes[-(MOMENTUM_DAYS + 1)] if len(closes) > MOMENTUM_DAYS else closes[0]
    return {
        "ticker":   ticker,
        "price":    price,
        "sma200":   sma,
        "above":    price > sma,
        "momentum": (price / past - 1) if past > 0 else 0.0,
    }


def pick_targets(api):
    """The ETFs we should be holding right now. Empty list means: be in cash."""
    reads = [r for r in (evaluate(api, t) for t in UNIVERSE) if r]
    for r in reads:
        print(f"  {r['ticker']}: ${r['price']:.2f} vs 200d ${r['sma200']:.2f} "
              f"({'ABOVE' if r['above'] else 'below'})  6mo {r['momentum']*100:+.1f}%")
    uptrends = [r for r in reads if r["above"]]
    uptrends.sort(key=lambda r: r["momentum"], reverse=True)
    return uptrends[:MAX_HOLDINGS], reads


def rebalance(api, dry_run=False):
    """
    Compare what we hold to what we should hold, and fix the difference.
    Returns a human-readable list of the actions taken.
    """
    targets, reads = pick_targets(api)
    target_syms    = [t["ticker"] for t in targets]

    try:
        positions = {p.symbol: p for p in api.list_positions()}
    except Exception as e:
        print(f"  rebalance: could not read positions: {e}")
        return []

    actions = []

    # 1. Sell anything we hold that is no longer a target (trend broke, or a
    #    stronger uptrend displaced it).
    for sym, pos in positions.items():
        if sym not in target_syms:
            qty = int(float(pos.qty))
            if qty <= 0:
                continue
            why = "trend broke" if sym in UNIVERSE else "not in universe"
            actions.append(f"SELL {qty} {sym} ({why})")
            if not dry_run:
                try:
                    api.submit_order(symbol=sym, qty=qty, side="sell",
                                     type="market", time_in_force="day")
                except Exception as e:
                    print(f"  sell {sym} failed: {e}")

    # 2. Buy anything that is a target but not held, sized equally.
    missing = [t for t in targets if t["ticker"] not in positions]
    if missing:
        try:
            equity = float(api.get_account().equity)
        except Exception as e:
            print(f"  rebalance: could not read equity: {e}")
            return actions
        alloc = (equity * INVEST_PCT) / max(len(targets), 1)
        for t in missing:
            qty = int(alloc / t["price"])
            if qty < 1:
                continue
            actions.append(f"BUY {qty} {t['ticker']} (above 200d, "
                           f"6mo {t['momentum']*100:+.1f}%)")
            if not dry_run:
                try:
                    api.submit_order(symbol=t["ticker"], qty=qty, side="buy",
                                     type="market", time_in_force="day")
                except Exception as e:
                    print(f"  buy {t['ticker']} failed: {e}")

    if not actions:
        held = ", ".join(sorted(positions)) or "cash"
        print(f"  No change needed — holding {held}")
    return actions


def status(api):
    """Snapshot for the dashboard."""
    targets, reads = pick_targets(api)
    return {
        "universe": reads,
        "targets":  [t["ticker"] for t in targets],
        "in_cash":  len(targets) == 0,
    }
