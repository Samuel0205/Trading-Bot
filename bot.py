# ── Market regime ─────────────────────────────────────────────
def update_market_regime():
    global market_regime  # <--- FIXED: moved inside the function
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=10)
        bars  = api.get_bars("SPY", "1Day",
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=10).df
        if bars.empty or len(bars) < 5:
            print("Regime: not enough SPY data")
            return
        if hasattr(bars.index, 'levels'):
            if "SPY" in bars.index.get_level_values(0):
                bars = bars.loc["SPY"]
        closes = list(bars["close"])
        ma5    = calc_ma(closes, 5)
        ma10   = calc_ma(closes, min(10, len(closes)))
        latest = closes[-1]
        if   ma5 > ma10*1.005 and latest > ma5: market_regime = "trending_up"
        elif ma5 < ma10*0.995 and latest < ma5: market_regime = "trending_down"
        else:                                    market_regime = "ranging"
        print(f"Regime: {market_regime} (SPY ${latest:.2f})")
    except Exception as e:
        print(f"Regime error: {e}")

def update_market_regime_with_retry(attempts=5, delay=10):
    global market_regime  # <--- FIXED: moved inside the function
    for i in range(attempts):
        update_market_regime()
        if market_regime != "unknown":
            return
        print(f"  Regime retry {i+1}/{attempts} in {delay}s...")
        time.sleep(delay)
    market_regime = "ranging"
    print("  Regime defaulting to: ranging")
