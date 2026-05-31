"""
database.py — SQLite persistence layer

Stores:
  - All trades (entry, exit, signals, P&L)
  - Signal performance over time
  - Daily portfolio snapshots
  - Macro event blackout dates
  - Scan results history
  - Alert log
  - Price history cache

Survives Render restarts. Self-initializes on first run.

FIXES vs original:
  - save_trade_close: SQLite doesn't support ORDER BY in UPDATE.
    Fixed using a subquery to find the correct row ID first.
  - Added indexes on trades(ticker), trades(entry_ts), alerts(acknowledged).
  - Each function creates its own connection (thread-safe for multi-thread Flask).
"""

import sqlite3, json, os, threading
from datetime import datetime, date
import pytz

DB_PATH = os.environ.get("DB_PATH", "/opt/render/project/src/trading_bot.db")
NY      = pytz.timezone("America/New_York")

# One lock per process — prevents "database is locked" under high thread load
_db_lock = threading.Lock()

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # WAL mode: readers don't block writers
    conn.execute("PRAGMA synchronous=NORMAL") # faster than FULL, safe enough
    return conn

def init_db():
    """Create all tables and indexes if they don't exist."""
    with _db_lock:
        conn = get_conn()
        c    = conn.cursor()

        # Trades table
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT NOT NULL,
                side          TEXT NOT NULL,
                qty           INTEGER NOT NULL,
                price         REAL NOT NULL,
                pnl           REAL,
                reason        TEXT,
                pred_score    REAL,
                rvol          REAL,
                is_gap        INTEGER DEFAULT 0,
                stop_price    REAL,
                target_price  REAL,
                active_signals TEXT,
                entry_ts      INTEGER,
                exit_ts       INTEGER,
                date          TEXT
            )
        """)

        # Signal performance
        c.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                was_win     INTEGER NOT NULL,
                ticker      TEXT,
                pnl         REAL,
                date        TEXT
            )
        """)

        # Portfolio snapshots
        c.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT UNIQUE,
                equity      REAL,
                cash        REAL,
                pnl_day     REAL,
                regime      TEXT,
                trade_count INTEGER DEFAULT 0
            )
        """)

        # Macro event blackout dates
        c.execute("""
            CREATE TABLE IF NOT EXISTS macro_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date  TEXT UNIQUE,
                event_name  TEXT,
                impact      TEXT DEFAULT 'high',
                source      TEXT
            )
        """)

        # Alerts log
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           INTEGER,
                level        TEXT,
                message      TEXT,
                ticker       TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        """)

        # Scan history
        c.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                date      TEXT,
                scan_time TEXT,
                tickers   TEXT,
                scores    TEXT
            )
        """)

        # Price history cache
        c.execute("""
            CREATE TABLE IF NOT EXISTS price_cache (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                price  REAL NOT NULL,
                volume REAL,
                ts     INTEGER,
                date   TEXT
            )
        """)

        # ── Indexes (performance) ─────────────────────────────
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker   ON trades(ticker)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts       ON trades(entry_ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_pnl      ON trades(pnl)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ack      ON alerts(acknowledged)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_macro_date      ON macro_events(event_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_price_cache     ON price_cache(ticker, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sigperf_name    ON signal_performance(signal_name)")

        conn.commit()
        conn.close()
    print("Database initialized:", DB_PATH)

# ── Trade operations ──────────────────────────────────────────

def save_trade_open(ticker, qty, price, stop, target, pred_score,
                    rvol, is_gap, active_signals, ts):
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO trades (ticker, side, qty, price, stop_price, target_price,
                                pred_score, rvol, is_gap, active_signals, entry_ts, date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ticker, "BUY", qty, price, stop, target, pred_score, rvol,
              1 if is_gap else 0, json.dumps(active_signals), ts,
              datetime.now(NY).strftime("%Y-%m-%d")))
        conn.commit()
        conn.close()

def save_trade_close(ticker, exit_price, pnl, reason, exit_ts):
    """
    FIX: SQLite does not support ORDER BY in UPDATE statements.
    We use a subquery to find the most recent open BUY row first,
    then update by its primary key id.
    """
    with _db_lock:
        conn = get_conn()
        try:
            # Find the most recent open (pnl IS NULL) BUY row for this ticker
            row = conn.execute("""
                SELECT id FROM trades
                WHERE ticker=? AND pnl IS NULL AND side='BUY'
                ORDER BY entry_ts DESC
                LIMIT 1
            """, (ticker,)).fetchone()

            if row:
                conn.execute("""
                    UPDATE trades
                    SET pnl=?, reason=?, exit_ts=?, side='SELL'
                    WHERE id=?
                """, (pnl, reason, exit_ts, row["id"]))
                conn.commit()
            else:
                print(f"  save_trade_close: no open BUY found for {ticker}")
        except Exception as e:
            print(f"  save_trade_close error {ticker}: {e}")
        finally:
            conn.close()

def get_recent_trades(days=30):
    with _db_lock:
        conn  = get_conn()
        cutoff = int((datetime.now().timestamp() - days * 86400) * 1000)
        rows  = conn.execute("""
            SELECT * FROM trades WHERE entry_ts > ?
            ORDER BY entry_ts DESC LIMIT 200
        """, (cutoff,)).fetchall()
        conn.close()
    return [dict(r) for r in rows]

def get_trade_stats_from_db():
    with _db_lock:
        conn = get_conn()
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE pnl IS NOT NULL"
        ).fetchall()
        conn.close()
    pnls   = [r["pnl"] for r in rows]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total":       len(pnls),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    round(len(wins) / max(len(pnls), 1) * 100, 1),
        "total_pnl":   round(sum(pnls), 2),
        "best_trade":  round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "avg_win":     round(sum(wins)   / max(len(wins),   1), 2),
        "avg_loss":    round(sum(losses) / max(len(losses), 1), 2),
    }

# ── Signal performance ────────────────────────────────────────

def save_signal_performance(signal_name, was_win, ticker, pnl):
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO signal_performance (signal_name, was_win, ticker, pnl, date)
            VALUES (?,?,?,?,?)
        """, (signal_name, 1 if was_win else 0, ticker, pnl,
              datetime.now(NY).strftime("%Y-%m-%d")))
        conn.commit()
        conn.close()

def get_signal_win_rates():
    with _db_lock:
        conn  = get_conn()
        rows  = conn.execute("""
            SELECT signal_name,
                   SUM(was_win) as wins,
                   COUNT(*)     as total
            FROM signal_performance
            GROUP BY signal_name
        """).fetchall()
        conn.close()
    result = {}
    for r in rows:
        total = r["total"]
        result[r["signal_name"]] = {
            "wins":     r["wins"],
            "total":    total,
            "win_rate": round(r["wins"] / max(total, 1) * 100, 1)
        }
    return result

# ── Portfolio snapshots ───────────────────────────────────────

def save_portfolio_snapshot(equity, cash, pnl_day, regime, trade_count):
    today = datetime.now(NY).strftime("%Y-%m-%d")
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO portfolio_snapshots (date, equity, cash, pnl_day, regime, trade_count)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET
                equity=excluded.equity, cash=excluded.cash,
                pnl_day=excluded.pnl_day, regime=excluded.regime,
                trade_count=excluded.trade_count
        """, (today, equity, cash, pnl_day, regime, trade_count))
        conn.commit()
        conn.close()

def get_portfolio_history(days=90):
    with _db_lock:
        conn = get_conn()
        rows = conn.execute("""
            SELECT * FROM portfolio_snapshots
            ORDER BY date DESC LIMIT ?
        """, (days,)).fetchall()
        conn.close()
    return [dict(r) for r in rows][::-1]

# ── Macro events ──────────────────────────────────────────────

def save_macro_event(event_date, event_name, impact="high", source="manual"):
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO macro_events (event_date, event_name, impact, source)
            VALUES (?,?,?,?)
            ON CONFLICT(event_date) DO UPDATE SET
                event_name=excluded.event_name, impact=excluded.impact
        """, (event_date, event_name, impact, source))
        conn.commit()
        conn.close()

def is_macro_blackout(check_date=None):
    """Returns (True, event_name) if date is a high-impact macro event day."""
    if check_date is None:
        check_date = datetime.now(NY).strftime("%Y-%m-%d")
    with _db_lock:
        conn = get_conn()
        row  = conn.execute("""
            SELECT * FROM macro_events WHERE event_date=? AND impact='high'
        """, (check_date,)).fetchone()
        conn.close()
    if row:
        return True, row["event_name"]
    return False, None

def get_upcoming_macro_events(days=7):
    today = datetime.now(NY).strftime("%Y-%m-%d")
    with _db_lock:
        conn = get_conn()
        rows = conn.execute("""
            SELECT * FROM macro_events
            WHERE event_date >= ? ORDER BY event_date LIMIT 20
        """, (today,)).fetchall()
        conn.close()
    return [dict(r) for r in rows]

# ── Alerts ────────────────────────────────────────────────────

def save_alert(level, message, ticker=None):
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO alerts (ts, level, message, ticker)
            VALUES (?,?,?,?)
        """, (int(datetime.now().timestamp() * 1000), level, message, ticker))
        conn.commit()
        conn.close()
    print(f"[{level}] {message}" + (f" ({ticker})" if ticker else ""))

def get_alerts(limit=50, unacknowledged_only=False):
    with _db_lock:
        conn = get_conn()
        q    = "SELECT * FROM alerts"
        if unacknowledged_only:
            q += " WHERE acknowledged=0"
        q   += " ORDER BY ts DESC LIMIT ?"
        rows = conn.execute(q, (limit,)).fetchall()
        conn.close()
    return [dict(r) for r in rows]

def acknowledge_alerts():
    with _db_lock:
        conn = get_conn()
        conn.execute("UPDATE alerts SET acknowledged=1")
        conn.commit()
        conn.close()

# ── Price cache ───────────────────────────────────────────────

def save_price_history(ticker, prices, volumes):
    """Persist recent price/volume history so signals survive restarts."""
    today = datetime.now(NY).strftime("%Y-%m-%d")
    with _db_lock:
        conn = get_conn()
        conn.execute("DELETE FROM price_cache WHERE ticker=?", (ticker,))
        ts = int(datetime.now().timestamp() * 1000)
        for i, (p, v) in enumerate(zip(prices[-200:], volumes[-200:])):
            conn.execute("""
                INSERT INTO price_cache (ticker, price, volume, ts, date)
                VALUES (?,?,?,?,?)
            """, (ticker, p, v or 0, ts - (len(prices) - i) * 30000, today))
        conn.commit()
        conn.close()

def load_price_history(ticker):
    """Load persisted price history for a ticker."""
    with _db_lock:
        conn  = get_conn()
        rows  = conn.execute("""
            SELECT price, volume FROM price_cache
            WHERE ticker=? ORDER BY ts ASC LIMIT 200
        """, (ticker,)).fetchall()
        conn.close()
    prices  = [r["price"]  for r in rows]
    volumes = [r["volume"] for r in rows]
    return prices, volumes

# ── Partial exit log ─────────────────────────────────────────

def save_partial_exit(ticker, qty, entry_price, exit_price, pnl, ts):
    """
    Record a partial exit (half-position sell at +6%) as a separate SELL row.
    Does NOT close the original BUY row — that stays open until the full exit.
    """
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO trades (ticker, side, qty, price, pnl, reason, entry_ts, date)
            VALUES (?,?,?,?,?,?,?,?)
        """, (ticker, "PARTIAL", qty, exit_price, pnl, "partial_exit", ts,
              datetime.now(NY).strftime("%Y-%m-%d")))
        conn.commit()
        conn.close()

# ── Scan history ──────────────────────────────────────────────

def save_scan_result(tickers, scores):
    now = datetime.now(NY)
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT INTO scan_history (date, scan_time, tickers, scores)
            VALUES (?,?,?,?)
        """, (now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
              json.dumps(tickers), json.dumps(scores)))
        conn.commit()
        conn.close()
