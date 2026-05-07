<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Backtest</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f1117; color: #e8e8e8; padding: 12px; font-size: 14px; }
.header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.header h1 { font-size: 18px; font-weight: 600; }
.back { font-size: 12px; color: #1D9E75; text-decoration: none; background: #1D9E7511; padding: 5px 10px; border-radius: 8px; }
.toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.run-time { font-size: 11px; color: #444; }
.run-btn { font-size: 12px; font-weight: 500; padding: 7px 16px; border-radius: 8px; border: 1px solid #1D9E7544; background: #1D9E7511; color: #1D9E75; cursor: pointer; }
.run-btn:disabled { opacity: .4; cursor: default; }
.status-msg { font-size: 11px; color: #E9A84C; margin-bottom: 8px; min-height: 16px; }
.tabs { display: flex; gap: 8px; margin-bottom: 14px; }
.tab { padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 500; background: #1a1d27; color: #888; cursor: pointer; border: none; }
.tab.active { background: #1D9E7522; color: #1D9E75; }
.section { display: none; }
.section.active { display: block; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
.stat-card { background: #1a1d27; border-radius: 10px; padding: 12px; }
.stat-card .lbl { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.stat-card .val { font-size: 20px; font-weight: 700; }
.up { color: #1D9E75; } .down { color: #D85A30; }
.chart-card { background: #1a1d27; border-radius: 10px; padding: 14px; margin-bottom: 14px; }
.chart-wrap { position: relative; width: 100%; height: 160px; }
.ticker-perf { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
.tp-row { background: #1a1d27; border-radius: 8px; padding: 10px 12px; display: flex; align-items: center; gap: 10px; }
.tp-rank { font-size: 11px; color: #555; min-width: 20px; }
.tp-sym { font-size: 13px; font-weight: 600; flex: 1; }
.tp-pnl { font-size: 13px; font-weight: 600; min-width: 60px; text-align: right; }
.tp-meta { font-size: 10px; color: #555; }
.trade-list { display: flex; flex-direction: column; gap: 6px; max-height: 400px; overflow-y: auto; }
.trade-row { background: #1a1d27; border-radius: 8px; padding: 10px 12px; font-size: 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.t-side { font-weight: 600; min-width: 32px; }
.t-pnl  { font-weight: 600; margin-left: auto; }
.t-reason { font-size: 10px; color: #555; }
.config-card { background: #1a1d27; border-radius: 10px; padding: 14px; margin-bottom: 14px; }
.config-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2a2d3a; font-size: 12px; }
.config-row:last-child { border-bottom: none; }
.config-row .lbl { color: #555; }
.config-row .val { color: #e8e8e8; font-weight: 500; }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.compare-card { background: #1a1d27; border-radius: 10px; padding: 12px; }
.compare-card .title { font-size: 11px; color: #555; margin-bottom: 8px; }
.compare-stat { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; }
.compare-stat .lk { color: #666; }
.compare-stat .vk { font-weight: 600; }
.empty { color: #444; font-size: 13px; text-align: center; padding: 48px 16px; line-height: 1.8; }
.loading { display: flex; align-items: center; gap: 10px; color: #555; font-size: 13px; padding: 40px 0; justify-content: center; }
.spinner { width: 16px; height: 16px; border: 2px solid #2a2d3a; border-top-color: #1D9E75; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="header">
  <h1>Backtest</h1>
  <a href="/" class="back">← Dashboard</a>
</div>

<div class="toolbar">
  <span id="runTime" class="run-time">Run a backtest to see results</span>
  <button id="runBtn" class="run-btn" onclick="startBacktest()">Run backtest</button>
</div>
<div class="status-msg" id="statusMsg"></div>

<div class="tabs" id="tabs" style="display:none;">
  <button class="tab active" onclick="showTab('overview')">Overview</button>
  <button class="tab" onclick="showTab('tickers')">By ticker</button>
  <button class="tab" onclick="showTab('trades')">Trades</button>
  <button class="tab" onclick="showTab('compare')">Compare</button>
  <button class="tab" onclick="showTab('config')">Config</button>
</div>

<div id="content">
  <div class="empty">No backtest data yet.<br>Tap "Run backtest" to replay 3 months of daily data<br>through the exact same signals the live bot uses.</div>
</div>

<script>
let lastResult = null;
let activeTabName = 'overview';
let equityChart = null;

function showTab(name) {
  activeTabName = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => { if(t.textContent.toLowerCase().includes(name.slice(0,3).toLowerCase())) t.classList.add('active'); });
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('sec-'+name);
  if (el) el.classList.add('active');
  // If clicking the compare tab, show comparison; otherwise show first universe by default
  if (name === 'overview' || name === 'tickers' || name === 'trades') {
    const names = Object.keys(lastResult?.results || {});
    if (names.length) renderUniverse(names[0]);
  }
}

function pct(v) { return (v >= 0 ? '+' : '') + v + '%'; }
function dollar(v) { return (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2); }
function colorClass(v) { return v >= 0 ? 'up' : 'down'; }

function renderUniverse(name) {
  if (!lastResult || !lastResult.results[name]) return;
  const u  = lastResult.results[name];
  const s  = u.stats;
  const ec = u.equity_curve || [];

  // Stats grid
  document.getElementById('stats-grid').innerHTML = [
    { lbl:'Total return', val: dollar(s.total_return), cls: colorClass(s.total_return) },
    { lbl:'Return %',     val: pct(s.return_pct),      cls: colorClass(s.return_pct) },
    { lbl:'Win rate',     val: s.win_rate+'%',          cls: s.win_rate >= 50 ? 'up' : 'down' },
    { lbl:'Total trades', val: s.total_trades,          cls: '' },
    { lbl:'Sharpe ratio', val: s.sharpe,                cls: s.sharpe >= 1 ? 'up' : 'down' },
    { lbl:'Max drawdown', val: dollar(-Math.abs(s.max_drawdown)), cls: 'down' },
    { lbl:'Best trade',   val: dollar(s.best_trade),    cls: 'up' },
    { lbl:'Worst trade',  val: dollar(s.worst_trade),   cls: 'down' },
    { lbl:'Avg hold',     val: s.avg_hold_bars+' days', cls: '' },
    { lbl:'Wins/Losses',  val: s.wins+' / '+s.losses,   cls: '' },
  ].map(c => `<div class="stat-card"><div class="lbl">${c.lbl}</div><div class="val ${c.cls}">${c.val}</div></div>`).join('');

  // Equity curve
  if (equityChart) { equityChart.destroy(); equityChart = null; }
  const labels = ec.map((_, i) => 'D'+i);
  const color  = ec.length && ec[ec.length-1] >= ec[0] ? '#1D9E75' : '#D85A30';
  equityChart = new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{ data: ec, borderColor: color, borderWidth: 2, fill: true,
        backgroundColor: color+'18', pointRadius: 0, tension: 0.3 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { ticks: { color:'#555', font:{size:11}, callback: v=>'$'+v.toFixed(2) }, grid:{color:'#1e2130'} },
        x: { ticks: { color:'#444', font:{size:9}, maxTicksLimit:8 }, grid:{display:false} }
      }
    }
  });

  // Ticker performance
  const tp = u.tickers_tested || [];
  document.getElementById('ticker-perf').innerHTML = tp.map((t, i) => {
    const sign = t.pnl >= 0 ? '+' : '';
    return `<div class="tp-row">
      <span class="tp-rank">${i+1}</span>
      <span class="tp-sym">${t.ticker}</span>
      <span class="tp-meta">${t.trades} trades · ${t.win_rate}% wr</span>
      <span class="tp-pnl ${colorClass(t.pnl)}">${sign}$${Math.abs(t.pnl).toFixed(2)}</span>
    </div>`;
  }).join('') || '<div style="color:#555;font-size:12px;padding:8px;">No ticker data</div>';

  // Trade list
  const trades = u.trades || [];
  document.getElementById('trade-list').innerHTML = trades.slice(0, 80).map(t => {
    const pnlSign = t.pnl >= 0 ? '+' : '';
    const entPct  = ((t.exit - t.entry) / t.entry * 100).toFixed(1);
    return `<div class="trade-row">
      <span class="t-side ${t.pnl>=0?'up':'down'}">${t.ticker}</span>
      <span style="color:#555;font-size:10px;">${t.date_entry||''}</span>
      <span style="color:#888;">$${t.entry} → $${t.exit}</span>
      <span class="t-reason">${t.reason||''} · ${t.bars_held}d</span>
      <span class="t-pnl ${colorClass(t.pnl)}">${pnlSign}$${Math.abs(t.pnl).toFixed(2)} (${entPct}%)</span>
    </div>`;
  }).join('') || '<div style="color:#555;font-size:12px;padding:8px;">No trades</div>';
}

function renderCompare() {
  if (!lastResult) return;
  const names = Object.keys(lastResult.results);
  document.getElementById('compare-grid').innerHTML = names.map(name => {
    const s = lastResult.results[name].stats;
    return `<div class="compare-card">
      <div class="title">${name}</div>
      <div class="compare-stat"><span class="lk">Return</span><span class="vk ${colorClass(s.total_return)}">${pct(s.return_pct)}</span></div>
      <div class="compare-stat"><span class="lk">Win rate</span><span class="vk">${s.win_rate}%</span></div>
      <div class="compare-stat"><span class="lk">Trades</span><span class="vk">${s.total_trades}</span></div>
      <div class="compare-stat"><span class="lk">Sharpe</span><span class="vk ${s.sharpe>=1?'up':'down'}">${s.sharpe}</span></div>
      <div class="compare-stat"><span class="lk">Max DD</span><span class="vk down">-$${Math.abs(s.max_drawdown).toFixed(2)}</span></div>
      <div class="compare-stat"><span class="lk">Avg hold</span><span class="vk">${s.avg_hold_bars}d</span></div>
    </div>`;
  }).join('');
}

function renderResult(data) {
  lastResult = data;
  if (!data || !data.results || !Object.keys(data.results).length) return;

  document.getElementById('tabs').style.display = 'flex';
  document.getElementById('runTime').textContent = 'Run at ' + (data.run_at || '—');

  const cfg = data.config || {};
  document.getElementById('config-rows').innerHTML = [
    { l:'Starting capital',  v:'$'+cfg.starting_capital },
    { l:'Months back',       v:cfg.months_back+' months' },
    { l:'Vote threshold',    v:cfg.threshold },
    { l:'Stop loss',         v:(cfg.stop_loss_pct*100).toFixed(0)+'%' },
    { l:'Take profit',       v:(cfg.take_profit_pct*100).toFixed(0)+'%' },
    { l:'Friction per trade',v:(cfg.friction_pct*100).toFixed(1)+'%' },
    { l:'Cooldown bars',     v:cfg.cooldown_bars+' days' },
  ].map(r=>`<div class="config-row"><span class="lbl">${r.l}</span><span class="val">${r.v}</span></div>`).join('');

  document.getElementById('content').innerHTML = `
    <div id="sec-overview" class="section active">
      <div class="stats-grid" id="stats-grid"></div>
      <div class="chart-card">
        <div style="font-size:11px;color:#555;margin-bottom:10px;text-transform:uppercase;">Equity curve</div>
        <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
      </div>
    </div>
    <div id="sec-tickers" class="section">
      <div class="ticker-perf" id="ticker-perf"></div>
    </div>
    <div id="sec-trades" class="section">
      <div class="trade-list" id="trade-list"></div>
    </div>
    <div id="sec-compare" class="section">
      <div class="compare-grid" id="compare-grid"></div>
    </div>
    <div id="sec-config" class="section">
      <div class="config-card" id="config-rows"></div>
    </div>
  `;

  const names = Object.keys(data.results);
  if (names.length) renderUniverse(names[0]);
  renderCompare();
  document.getElementById('tabs').querySelectorAll('.tab').forEach((t,i)=>{
    t.onclick = () => {
      const tabNames = ['overview','tickers','trades','compare','config'];
      showTab(tabNames[i]);
    };
  });
}

function startBacktest() {
  const btn = document.getElementById('runBtn');
  btn.textContent = 'Running...'; btn.disabled = true;
  document.getElementById('statusMsg').textContent = 'Fetching 3 months of bar data...';
  document.getElementById('content').innerHTML =
    '<div class="loading"><div class="spinner"></div>Running backtest — takes 1–2 minutes...</div>';
  document.getElementById('tabs').style.display = 'none';
  fetch('/backtest/run', { method: 'POST' })
    .catch(e => {
      console.error(e);
      document.getElementById('statusMsg').textContent = 'Error starting backtest';
      btn.textContent = 'Run backtest'; btn.disabled = false;
    });
  setTimeout(() => { if (btn.disabled) { btn.textContent = 'Run backtest'; btn.disabled = false; } }, 180000);
}

// Load existing results
fetch('/backtest/data').then(r=>r.json()).then(data=>{
  if (data && data.results && Object.keys(data.results).length) renderResult(data);
}).catch(()=>{});

const socket = io();
socket.on('backtest_result', data => {
  document.getElementById('runBtn').textContent = 'Run backtest';
  document.getElementById('runBtn').disabled = false;
  document.getElementById('statusMsg').textContent = '';
  renderResult(data);
});
socket.on('backtest_status', data => {
  const msg = document.getElementById('statusMsg');
  if (data.status === 'running') {
    msg.textContent = data.message || 'Running...';
  } else if (data.status === 'error') {
    msg.textContent = 'Error: ' + (data.message || 'unknown');
    msg.style.color = '#D85A30';
    document.getElementById('runBtn').textContent = 'Run backtest';
    document.getElementById('runBtn').disabled = false;
  } else if (data.status === 'done') {
    msg.textContent = '';
  }
});
</script>
</body>
</html>
