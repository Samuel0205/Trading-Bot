"""
ops_reviewer.py — Self-monitoring for daily operational events.

bot.py populates _ops_stats throughout the trading day.
build_ops_findings() runs rule-based checks and returns a list of finding dicts.
analyze_with_claude() calls the Anthropic API for deeper AI analysis, cached 1 hour.
"""

import os
import re
import json
import time

# ── Shared state (bot.py writes here) ────────────────────────────────────────
_ops_stats = {
    "date":                 None,
    "daily_limit_hits":     {},     # {ticker: count}
    "filter_blocks":        {},     # {reason_str: count}
    "finbert_errors":       0,      # count of keyword_fallback responses
    "finbert_method":       None,   # last method: "finbert", "keyword", "keyword_fallback"
    "scan_tickers":         [],     # tickers from last scan
    "blocked_buy_tickers":  set(),  # tickers that wanted to buy but were blocked
}

# ── Claude result cache ───────────────────────────────────────────────────────
_ai_cache = {"ts": 0.0, "result": None}
_AI_CACHE_TTL = 3600  # 1 hour


# ── Rule-based findings ───────────────────────────────────────────────────────

def build_ops_findings(ops_stats, rvol_cache, prediction_cache, active_tickers, regime):
    """
    Evaluate operational state and return a list of finding dicts:
      [{priority, area, finding, action}, ...]

    Only emits a finding when its condition is triggered.
    """
    findings = []

    # ── Daily limit hits ──────────────────────────────────────────────────────
    try:
        limit_hits = ops_stats.get("daily_limit_hits", {})
        total_hits = sum(limit_hits.values())
        if total_hits > 3:
            tickers_str = ", ".join(
                f"{t}×{c}" for t, c in sorted(limit_hits.items(), key=lambda x: -x[1])
            )
            findings.append({
                "priority": "HIGH",
                "area":     "Trade Blocking",
                "finding":  f"Daily trade limit hit {total_hits} times today ({tickers_str}). "
                            "The bot is being blocked from entering new positions.",
                "action":   "Check MAX_DAILY_TRADES in bot.py — consider raising it if the "
                            "market conditions warrant more trades.",
            })
    except Exception:
        pass

    # ── Price range filter blocks ─────────────────────────────────────────────
    try:
        price_blocks = ops_stats.get("filter_blocks", {}).get("price_range", 0)
        if price_blocks > 3:
            findings.append({
                "priority": "MEDIUM",
                "area":     "Price Filter",
                "finding":  f"{price_blocks} stocks rejected today due to price range filter.",
                "action":   "Stocks are being screened out on price. Review get_price_floor() "
                            "and get_price_ceiling() thresholds in bot.py.",
            })
    except Exception:
        pass

    # ── RVOL filter blocks ────────────────────────────────────────────────────
    try:
        rvol_blocks = ops_stats.get("filter_blocks", {}).get("rvol", 0)
        if rvol_blocks > 5:
            findings.append({
                "priority": "LOW",
                "area":     "RVOL Filter",
                "finding":  f"{rvol_blocks} buy attempts blocked by RVOL filter today.",
                "action":   "High RVOL block count is expected with the IEX free data feed. "
                            "RVOL_THRESHOLD in bot.py gates entries on thin volume.",
            })
    except Exception:
        pass

    # ── RVOL zero-value ratio ─────────────────────────────────────────────────
    try:
        if rvol_cache:
            zero_count = sum(1 for v in rvol_cache.values() if not v)
            zero_pct = zero_count / len(rvol_cache) * 100
            if zero_pct > 70:
                findings.append({
                    "priority": "MEDIUM",
                    "area":     "Volume Data",
                    "finding":  f"{zero_pct:.0f}% of tracked tickers have zero/null RVOL "
                                f"({zero_count}/{len(rvol_cache)} tickers).",
                    "action":   "IEX free tier has limited volume coverage. Consider upgrading "
                                "to SIP feed or accepting zero-rvol tickers without blocking.",
                })
    except Exception:
        pass

    # ── FinBERT method tracking ───────────────────────────────────────────────
    try:
        # Read finbert_client._state directly instead of modifying predictions.py
        import finbert_client
        method = finbert_client._state.get("reachable")
        fb_method = ops_stats.get("finbert_method")

        # Check last known method from _ops_stats; fall back to finbert_client state
        if fb_method is None:
            # Infer from finbert_client state
            if not finbert_client.HF_TOKEN:
                fb_method = "keyword"
            elif not finbert_client._state.get("reachable", True):
                fb_method = "keyword_fallback"

        if fb_method == "keyword_fallback":
            findings.append({
                "priority": "HIGH",
                "area":     "FinBERT",
                "finding":  "FinBERT API is failing — falling back to keyword sentiment scoring. "
                            "Sentiment quality is degraded.",
                "action":   "Check HuggingFace API status and verify HUGGINGFACE_TOKEN is valid. "
                            "The router URL may have changed.",
            })
        elif fb_method == "keyword" and not finbert_client.HF_TOKEN:
            findings.append({
                "priority": "LOW",
                "area":     "FinBERT",
                "finding":  "FinBERT token not configured — using keyword sentiment only.",
                "action":   "Set HUGGINGFACE_TOKEN in Render environment variables to enable "
                            "FinBERT for better news sentiment scoring.",
            })
    except Exception:
        pass

    # ── Blocked buy tickers ───────────────────────────────────────────────────
    try:
        blocked = ops_stats.get("blocked_buy_tickers", set())
        if blocked:
            tickers_str = ", ".join(sorted(blocked))
            findings.append({
                "priority": "MEDIUM",
                "area":     "Missed Signals",
                "finding":  f"Tickers signaled buy but were blocked by filters today: {tickers_str}.",
                "action":   "Review filter thresholds for these tickers. They may be viable "
                            "candidates that are being rejected by overly strict criteria.",
            })
    except Exception:
        pass

    # ── All prediction scores are zero ────────────────────────────────────────
    try:
        if prediction_cache:
            scores = [v.get("score", 0) for v in prediction_cache.values()
                      if isinstance(v, dict)]
            if scores and all(s == 0 for s in scores):
                findings.append({
                    "priority": "MEDIUM",
                    "area":     "Predictions",
                    "finding":  f"All {len(scores)} prediction scores are 0 — the prediction "
                                "loop may not have run yet today.",
                    "action":   "Check that prediction_loop is running (see /health). Predictions "
                                "run at PRED_HOURS in bot.py (default 9–14h ET).",
                })
    except Exception:
        pass

    return findings


# ── Claude AI analysis ────────────────────────────────────────────────────────

def analyze_with_claude(ops_snapshot, trade_summary):
    """
    Call Claude Haiku to analyze ops_snapshot + trade_summary.
    Returns a list of finding dicts, or None if unavailable/key missing.
    Caches result for 1 hour.
    """
    global _ai_cache

    # Check prerequisites
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic  # noqa: F401 — test import
    except ImportError:
        return None

    # Return cached result if still fresh
    if time.time() - _ai_cache["ts"] < _AI_CACHE_TTL and _ai_cache["result"] is not None:
        return _ai_cache["result"]

    try:
        client = anthropic.Anthropic(api_key=api_key)

        prompt_data = {
            "ops_snapshot": ops_snapshot,
            "trade_summary": trade_summary,
        }

        system_prompt = (
            "You are a trading bot operations analyst. "
            "Given operational metrics and trade performance data, identify actionable issues. "
            "Respond ONLY with a JSON array of findings. Each finding must have exactly these keys: "
            "priority (HIGH/MEDIUM/LOW), area (short label), finding (1-2 sentences), "
            "action (concrete recommendation). "
            "Return between 1 and 5 findings. Focus on the most impactful issues. "
            "Do not include any text outside the JSON array."
        )

        user_prompt = (
            "Analyze this trading bot's operational state and recent trade performance. "
            "Identify the top issues and what to do about them.\n\n"
            f"Data:\n{json.dumps(prompt_data, indent=2, default=str)}"
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )

        raw = message.content[0].text if message.content else ""

        # Extract JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return None

        findings = json.loads(match.group(0))
        if not isinstance(findings, list):
            return None

        # Validate each finding has required keys
        validated = []
        for f in findings:
            if isinstance(f, dict) and all(k in f for k in ("priority", "area", "finding", "action")):
                validated.append(f)

        _ai_cache = {"ts": time.time(), "result": validated}
        return validated

    except Exception as e:
        print(f"  ops_reviewer: Claude API error: {e}")
        return None
