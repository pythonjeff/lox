// LOX FUND Dashboard v4 - Performance-Only

// ============================================
// UTILITIES
// ============================================

function formatCurrency(value, decimals = 0) {
    if (Math.abs(value) < 100) decimals = 2;
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    }).format(value);
}

function formatPercent(value, decimals = 2) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(decimals)}%`;
}

function formatTime() {
    return new Date().toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function formatDateTime() {
    return new Date().toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// ============================================
// CONNECTION STATE & RETRY ENGINE
// ============================================

const connectionState = {
    lastPositionsUpdate: null,
    lastTradesUpdate: null,
    positionsRetries: 0,
    tradesRetries: 0,
    maxRetries: 5,
    baseDelay: 5000,      // 5s initial retry
    connected: false,
};

function getRetryDelay(retries) {
    // Exponential backoff: 5s, 10s, 20s, 40s, 60s max
    return Math.min(connectionState.baseDelay * Math.pow(2, retries), 60000);
}

function updateConnectionStatus(connected) {
    connectionState.connected = connected;
    const dot = document.getElementById('live-dot');
    const label = document.getElementById('live-label');
    if (!dot || !label) return;

    if (connected) {
        dot.className = 'live-dot';
        label.textContent = 'LIVE';
        label.className = '';
    } else {
        dot.className = 'live-dot live-dot--stale';
        label.textContent = 'STALE';
        label.className = 'stale-label';
    }
}

function updateStalenessCheck() {
    if (!connectionState.lastPositionsUpdate) return;
    const age = (Date.now() - connectionState.lastPositionsUpdate) / 1000;
    // If data is older than 2.5 minutes, mark as stale
    updateConnectionStatus(age < 150);
}

// ============================================
// RESILIENT FETCH WITH RETRY
// ============================================

async function fetchWithRetry(url, retryKey, processFunc) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // Reset retry counter on success
        connectionState[retryKey] = 0;
        updateConnectionStatus(true);
        processFunc(data);
        return data;
    } catch (err) {
        console.error(`[Dashboard] Fetch failed (${url}):`, err.message);
        connectionState[retryKey]++;

        if (connectionState[retryKey] <= connectionState.maxRetries) {
            const delay = getRetryDelay(connectionState[retryKey] - 1);
            console.log(`[Dashboard] Retry ${connectionState[retryKey]}/${connectionState.maxRetries} in ${delay/1000}s`);
            setTimeout(() => fetchWithRetry(url, retryKey, processFunc), delay);
        } else {
            updateConnectionStatus(false);
            console.warn(`[Dashboard] Max retries reached for ${url}`);
            // Reset retries after cooldown so next poll cycle tries again
            setTimeout(() => { connectionState[retryKey] = 0; }, 60000);
        }
        return null;
    }
}


// ============================================
// POSITIONS DATA (with AUM)
// ============================================
function processPositionsData(data) {
    if (data.error) {
        console.error('Positions API Error:', data.error);
    }

    connectionState.lastPositionsUpdate = Date.now();

    // Update header timestamp
    document.getElementById('header-timestamp').textContent = formatDateTime();

    // HERO: Fund Return
    const heroReturn = document.getElementById('hero-return');
    if (data.return_pct !== undefined) {
        const val = data.return_pct;
        heroReturn.textContent = formatPercent(val);
        heroReturn.className = 'hero-value ' + (val >= 0 ? 'positive' : 'negative');
    }

    // HERO: Benchmark comparison cards
    const sp500El = document.getElementById('bench-sp500');
    const macroEl = document.getElementById('bench-macro-hf');
    const alphaEl = document.getElementById('bench-alpha');
    if (sp500El && data.sp500_return !== undefined && data.sp500_return !== null) {
        sp500El.textContent = formatPercent(data.sp500_return);
        sp500El.className = 'bench-value ' + (data.sp500_return >= 0 ? 'positive' : 'negative');
    }
    if (macroEl && data.macro_hf_return !== undefined && data.macro_hf_return !== null) {
        macroEl.textContent = formatPercent(data.macro_hf_return);
        macroEl.className = 'bench-value ' + (data.macro_hf_return >= 0 ? 'positive' : 'negative');
    }
    if (alphaEl && data.alpha_sp500 !== undefined && data.alpha_sp500 !== null) {
        alphaEl.textContent = formatPercent(data.alpha_sp500);
        alphaEl.className = 'bench-value ' + (data.alpha_sp500 >= 0 ? 'positive' : 'negative');
    }


    // METRICS: NAV
    const navEl = document.getElementById('nav-value');
    navEl.textContent = data.nav_equity ? formatCurrency(data.nav_equity) : '—';

    // METRICS: P&L
    const pnlEl = document.getElementById('total-pnl');
    const navEquity = data.nav_equity || 0;
    const originalCapital = data.original_capital || data.aum || 3150;
    let pnl = navEquity - originalCapital;
    if (data.total_pnl !== null && data.total_pnl !== undefined) {
        pnl = data.total_pnl;
    }
    pnlEl.textContent = formatCurrency(pnl);
    pnlEl.className = 'metric-value ' + (pnl < 0 ? 'negative' : 'positive');

    // METRICS: Cash
    const cashEl = document.getElementById('cash-value');
    cashEl.textContent = data.cash_available ? formatCurrency(data.cash_available) : '—';

    // MOBILE HERO: Big NAV + P&L subtitle
    const mobileNav = document.getElementById('hero-mobile-nav');
    const mobilePnl = document.getElementById('hero-mobile-pnl');
    const mobilePnlPct = document.getElementById('hero-mobile-pnl-pct');
    if (mobileNav) {
        mobileNav.textContent = data.nav_equity ? formatCurrency(data.nav_equity) : '—';
    }
    if (mobilePnl) {
        const pnlSign = pnl >= 0 ? '+' : '';
        mobilePnl.textContent = `${pnlSign}${formatCurrency(pnl)}`;
        mobilePnl.className = 'hero-mobile-pnl-val ' + (pnl >= 0 ? 'positive' : 'negative');
    }
    if (mobilePnlPct && data.return_pct !== undefined) {
        const rp = data.return_pct;
        mobilePnlPct.textContent = `(${formatPercent(rp)})`;
        mobilePnlPct.className = 'hero-mobile-pnl-pct ' + (rp >= 0 ? 'positive' : 'negative');
    }

    // POSITIONS: Count badge
    const countEl = document.getElementById('positions-count');
    countEl.textContent = data.positions ? `${data.positions.length} POSITIONS` : '—';

    // POSITIONS: Unrealized P&L badge (muted styling)
    const posPnlEl = document.getElementById('positions-pnl');
    if (data.positions && data.positions.length > 0) {
        let totalPnl = 0;
        data.positions.forEach(p => { totalPnl += p.pnl || 0; });
        posPnlEl.textContent = formatCurrency(totalPnl);
        posPnlEl.className = 'badge-pnl muted';
    }

    // POSITIONS: Render as professional table
    const positionsList = document.getElementById('positions-list');
    if (!data.positions || data.positions.length === 0) {
        positionsList.innerHTML = '<div class="loading">No open positions</div>';
    } else {
        // Sort: winners first, then by absolute P&L descending
        const sorted = [...data.positions].sort((a, b) => (b.pnl || 0) - (a.pnl || 0));

        const rows = sorted.map(pos => {
            const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';

            let ticker, typeLabel, expiry = '';
            if (pos.opt_info) {
                const optType = (pos.opt_info.opt_type || '').toUpperCase().startsWith('C') ? 'Call' : 'Put';
                ticker = pos.opt_info.underlying;
                typeLabel = `$${pos.opt_info.strike} ${optType}`;
                expiry = pos.opt_info.expiry || '';
            } else {
                ticker = pos.symbol;
                typeLabel = 'Equity';
            }

            const qty = pos.qty || 0;
            const mv = Math.abs(pos.market_value || 0);
            const costBasis = mv - (pos.pnl || 0);
            const pnlVal = pos.pnl || 0;
            const pnlPct = pos.pnl_pct || 0;
            const daysOpen = pos.days_open !== null && pos.days_open !== undefined ? pos.days_open : '—';

            return `
                <tr class="pos-row">
                    <td class="pos-td-ticker">
                        <span class="pos-ticker">${ticker}</span>
                        <span class="pos-type">${typeLabel}</span>
                    </td>
                    <td class="pos-td-expiry">${expiry}</td>
                    <td class="pos-td-qty">${qty > 0 ? '+' : ''}${qty}</td>
                    <td class="pos-td-cost">${formatCurrency(costBasis)}</td>
                    <td class="pos-td-mv">${formatCurrency(mv)}</td>
                    <td class="pos-td-pnl ${pnlClass}">${formatCurrency(pnlVal)}</td>
                    <td class="pos-td-pct ${pnlClass}">${formatPercent(pnlPct, 1)}</td>
                    <td class="pos-td-days">${daysOpen}</td>
                </tr>`;
        }).join('');

        positionsList.innerHTML = `
            <table class="pos-table">
                <thead>
                    <tr>
                        <th class="pos-th-ticker">Position</th>
                        <th class="pos-th-expiry">Expiry</th>
                        <th class="pos-th-qty">Qty</th>
                        <th class="pos-th-cost">Cost</th>
                        <th class="pos-th-mv">Mkt Val</th>
                        <th class="pos-th-pnl">P&L</th>
                        <th class="pos-th-pct">%</th>
                        <th class="pos-th-days">Days Open</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    // Update footer
    document.getElementById('footer-time').textContent = `Updated ${formatTime()}`;
}


// ============================================
// TRADE PERFORMANCE (Closed Trades)
// ============================================
function processClosedTradesData(data) {
    connectionState.lastTradesUpdate = Date.now();

    const statsEl = document.getElementById('trade-stats');
    const realizedPnlEl = document.getElementById('realized-pnl');

    if (data.trades && data.trades.length > 0) {
        const winRate = data.win_rate || 0;
        if (statsEl) statsEl.textContent = `${data.trades.length} TRADES · ${winRate.toFixed(0)}% WIN`;

        const totalPnl = data.total_pnl || 0;
        if (realizedPnlEl) {
            realizedPnlEl.textContent = formatCurrency(totalPnl);
            realizedPnlEl.className = 'badge-pnl ' + (totalPnl < 0 ? 'negative' : 'positive');
        }

        // Render professional metrics panel
        renderPerformanceMetrics(data.metrics, data.trades.length, winRate);
    } else {
        if (statsEl) statsEl.textContent = 'NO CLOSED TRADES';
        if (realizedPnlEl) {
            realizedPnlEl.textContent = '$0';
            realizedPnlEl.className = 'badge-pnl';
        }
        // Hide metrics panel when no trades
        const metricsPanel = document.getElementById('performance-metrics');
        if (metricsPanel) metricsPanel.style.display = 'none';
    }

    // Trades table
    const tbody = document.getElementById('trades-body');
    if (!tbody) return;

    if (!data.trades || data.trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">No closed trades yet</td></tr>';
    } else {
        tbody.innerHTML = data.trades.map(trade => {
            const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
            const statusClass = trade.fully_closed ? 'closed' : 'partial';
            const statusText = trade.fully_closed ? 'CLOSED' : 'PARTIAL';

            return `
                <tr>
                    <td>
                        <span class="trade-symbol">${trade.symbol}</span>
                        <span class="trade-status ${statusClass}">${statusText}</span>
                    </td>
                    <td>${formatCurrency(trade.cost)}</td>
                    <td>${formatCurrency(trade.proceeds)}</td>
                    <td class="pnl-cell ${pnlClass}">
                        ${formatCurrency(trade.pnl)}
                        <span class="pnl-percent">${formatPercent(trade.pnl_pct, 1)}</span>
                    </td>
                </tr>
            `;
        }).join('');
    }

}

// Render institutional-grade performance metrics
function renderPerformanceMetrics(metrics, tradeCount, winRate) {
    if (!metrics) return;

    const panel = document.getElementById('performance-metrics');
    if (!panel) return;
    panel.style.display = 'block';

    // =========================================
    // SAMPLE DISCLOSURE BANNER
    // =========================================
    const disclosureEl = document.getElementById('sample-disclosure');
    const warningEl = document.getElementById('sample-warning');
    const rangeEl = document.getElementById('date-range');

    if (disclosureEl && metrics.sample_warning) {
        disclosureEl.style.display = 'flex';
        if (warningEl) warningEl.textContent = metrics.sample_warning;
        if (rangeEl && metrics.date_range) {
            rangeEl.textContent = metrics.date_range;
        }
    } else if (disclosureEl) {
        // Still show date range even without warning
        if (metrics.date_range) {
            disclosureEl.style.display = 'flex';
            disclosureEl.style.background = 'var(--bg-muted)';
            disclosureEl.style.borderColor = 'var(--border)';
            if (warningEl) warningEl.textContent = `${tradeCount} trades`;
            warningEl.style.color = 'var(--text-secondary)';
            if (rangeEl) rangeEl.textContent = metrics.date_range;
        } else {
            disclosureEl.style.display = 'none';
        }
    }

    // =========================================
    // OVERALL GRADE
    // =========================================
    const gradeBox = document.getElementById('perf-grade-box');
    const gradeEl = document.getElementById('perf-grade');
    if (gradeEl) gradeEl.textContent = metrics.overall_grade;
    if (gradeBox) {
        gradeBox.className = 'perf-grade-box grade-' + metrics.overall_grade.toLowerCase();
    }

    // =========================================
    // CORE STATS (Top row)
    // =========================================

    // Portfolio Sharpe (Annualized) - PRIMARY metric
    const sharpeEl = document.getElementById('portfolio-sharpe');
    if (sharpeEl) {
        if (metrics.portfolio_sharpe !== null && metrics.portfolio_sharpe !== undefined) {
            const s = metrics.portfolio_sharpe;
            sharpeEl.textContent = s.toFixed(2);
            sharpeEl.className = 'perf-stat-value' + (s >= 2 ? ' excellent' : s >= 1 ? ' positive' : s < 0 ? ' negative' : '');
        } else {
            sharpeEl.textContent = 'N/A';
            sharpeEl.className = 'perf-stat-value';
        }
    }

    // Profit Factor
    const pfEl = document.getElementById('profit-factor');
    if (pfEl) {
        const pf = metrics.profit_factor;
        pfEl.textContent = pf >= 999 ? '∞' : pf.toFixed(2);
        pfEl.className = 'perf-stat-value' + (pf >= 2 ? ' excellent' : pf >= 1.5 ? ' positive' : pf < 1 ? ' negative' : '');
    }

    // Expectancy
    const expEl = document.getElementById('expectancy');
    if (expEl) {
        const exp = metrics.expectancy;
        expEl.textContent = (exp >= 0 ? '+' : '') + formatCurrency(exp);
        expEl.className = 'perf-stat-value' + (exp >= 0 ? ' positive' : ' negative');
    }

    // Max Drawdown
    const ddEl = document.getElementById('max-drawdown');
    if (ddEl) {
        const dd = metrics.max_drawdown_pct || 0;
        if (dd > 0) {
            ddEl.textContent = '-' + dd.toFixed(1) + '%';
            ddEl.className = 'perf-stat-value' + (dd >= 25 ? ' negative' : dd >= 15 ? '' : ' positive');
        } else {
            ddEl.textContent = '0%';
            ddEl.className = 'perf-stat-value positive';
        }
    }

    // =========================================
    // PAYOFF ANALYSIS CARD
    // =========================================
    const avgWinEl = document.getElementById('avg-win');
    if (avgWinEl) avgWinEl.textContent = formatCurrency(metrics.avg_win) + ` (${metrics.avg_win_pct.toFixed(0)}%)`;

    const avgLossEl = document.getElementById('avg-loss');
    if (avgLossEl) avgLossEl.textContent = formatCurrency(metrics.avg_loss) + ` (${metrics.avg_loss_pct.toFixed(0)}%)`;

    const payoffEl = document.getElementById('payoff-ratio');
    if (payoffEl) {
        const pr = metrics.payoff_ratio;
        payoffEl.textContent = pr >= 999 ? '∞' : pr.toFixed(2) + ':1';
        payoffEl.className = 'perf-row-value' + (pr >= 1.5 ? ' positive' : pr < 1 ? ' negative' : '');
    }

    // =========================================
    // DISTRIBUTION CARD
    // =========================================
    const stdEl = document.getElementById('pnl-std');
    if (stdEl) {
        stdEl.textContent = metrics.pnl_pct_std.toFixed(1) + '%';
    }

    const skewEl = document.getElementById('skewness');
    if (skewEl) {
        const sk = metrics.skewness;
        let skewText = sk.toFixed(2);
        if (sk > 0.5) skewText += ' (right tail)';
        else if (sk < -0.5) skewText += ' (left tail)';
        skewEl.textContent = skewText;
        skewEl.className = 'perf-row-value' + (sk > 0 ? ' positive' : sk < -0.5 ? ' negative' : '');
    }

    const holdEl = document.getElementById('avg-holding');
    if (holdEl) {
        if (metrics.avg_holding_days !== null && metrics.avg_holding_days !== undefined) {
            holdEl.textContent = metrics.avg_holding_days.toFixed(0) + ' days';
        } else {
            holdEl.textContent = 'N/A';
        }
    }

    // =========================================
    // EXTREMES & RISK CARD
    // =========================================
    const lgWinEl = document.getElementById('largest-win');
    if (lgWinEl) lgWinEl.textContent = formatCurrency(metrics.largest_win) + ` (${metrics.largest_win_pct.toFixed(0)}%)`;

    const lgLossEl = document.getElementById('largest-loss');
    if (lgLossEl) lgLossEl.textContent = formatCurrency(metrics.largest_loss) + ` (${metrics.largest_loss_pct.toFixed(0)}%)`;

    const recoveryEl = document.getElementById('recovery-days');
    if (recoveryEl) {
        if (metrics.recovery_days !== null && metrics.recovery_days !== undefined) {
            recoveryEl.textContent = metrics.recovery_days + ' days';
        } else if (metrics.max_drawdown > 0) {
            recoveryEl.textContent = 'Not yet';
            recoveryEl.className = 'perf-row-value negative';
        } else {
            recoveryEl.textContent = 'N/A';
        }
    }

    // =========================================
    // SECONDARY METRICS ROW
    // =========================================
    const rEl = document.getElementById('r-multiple');
    if (rEl) {
        const r = metrics.r_multiple;
        rEl.textContent = (r >= 0 ? '+' : '') + r.toFixed(1) + 'R';
    }

    const kellyEl = document.getElementById('kelly-pct');
    if (kellyEl) {
        kellyEl.textContent = metrics.kelly_pct.toFixed(0) + '%';
    }

    const maxWinsEl = document.getElementById('max-wins');
    if (maxWinsEl) maxWinsEl.textContent = metrics.max_consec_wins;

    const maxLossesEl = document.getElementById('max-losses');
    if (maxLossesEl) maxLossesEl.textContent = metrics.max_consec_losses;
}


// ============================================
// EQUITY CURVE (Chart.js)
// ============================================

let equityCurveChart = null;

function processNavHistory(data, liveTwrPct) {
    const section = document.getElementById('equity-curve-section');
    const canvas = document.getElementById('equity-curve-chart');
    if (!section || !canvas || !data.series || data.series.length < 2) return;

    section.style.display = 'block';

    // Append live intraday point if available and newer than last snapshot
    const series = data.series.slice();
    if (liveTwrPct !== undefined && liveTwrPct !== null) {
        const today = new Date().toISOString().slice(0, 10);
        const lastDate = series[series.length - 1].date;
        if (today > lastDate) {
            series.push({ date: today, twr_cum_pct: liveTwrPct });
        } else if (today === lastDate) {
            series[series.length - 1].twr_cum_pct = liveTwrPct;
        }
    }

    const labels = series.map(p => {
        const d = new Date(p.date + 'T12:00:00');
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    const values = series.map(p => p.twr_cum_pct);

    const lastVal = values[values.length - 1];

    // Update chart subtitle with latest TWR
    const subtitleEl = document.getElementById('chart-latest');
    if (subtitleEl) {
        subtitleEl.textContent = (lastVal >= 0 ? '+' : '') + lastVal.toFixed(2) + '%';
        subtitleEl.className = 'chart-subtitle ' + (lastVal >= 0 ? 'positive' : 'negative');
    }

    const lineColor = lastVal >= 0 ? 'rgba(0, 168, 107, 1)' : 'rgba(230, 57, 70, 1)';
    const fillColor = lastVal >= 0 ? 'rgba(0, 168, 107, 0.08)' : 'rgba(230, 57, 70, 0.08)';

    if (equityCurveChart) equityCurveChart.destroy();

    equityCurveChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                borderColor: lineColor,
                backgroundColor: fillColor,
                fill: true,
                tension: 0.25,
                pointRadius: 0,
                pointHoverRadius: 3,
                pointHoverBackgroundColor: lineColor,
                borderWidth: 1.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { left: 0, right: 4, top: 4, bottom: 0 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleFont: { family: 'Inter', size: 10 },
                    bodyFont: { family: 'Inter', size: 11, weight: '600' },
                    padding: { x: 8, y: 4 },
                    cornerRadius: 2,
                    callbacks: {
                        label: function(ctx) {
                            const v = ctx.parsed.y;
                            return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: {
                        font: { size: 9, family: 'Inter' },
                        color: '#aaa',
                        maxTicksLimit: 6,
                        padding: 2,
                    }
                },
                y: {
                    grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
                    border: { display: false },
                    ticks: {
                        font: { size: 9, family: 'Inter' },
                        color: '#aaa',
                        padding: 4,
                        callback: function(value) {
                            return (value >= 0 ? '+' : '') + value.toFixed(0) + '%';
                        }
                    }
                }
            },
            interaction: { intersect: false, mode: 'index' },
        }
    });
}


// ============================================
// INIT & REFRESH
// ============================================

async function initDashboardParallel() {
    console.log('[Dashboard] Starting parallel load...');
    const startTime = performance.now();

    const fetches = [
        fetch('/api/positions').then(r => r.json()).catch(e => ({ error: e.message, positions: [] })),
        fetch('/api/closed-trades').then(r => r.json()).catch(e => ({ error: e.message, trades: [] })),
        fetch('/api/nav-history').then(r => r.json()).catch(e => ({ error: e.message, series: [] })),
    ];

    try {
        const [positionsData, tradesData, navData] = await Promise.all(fetches);

        processPositionsData(positionsData);
        processClosedTradesData(tradesData);
        processNavHistory(navData, positionsData.return_pct);

        const elapsed = (performance.now() - startTime).toFixed(0);
        console.log(`[Dashboard] Load complete in ${elapsed}ms`);
    } catch (err) {
        console.error('[Dashboard] Load error:', err);
        // Retry the full init after a short delay
        setTimeout(initDashboardParallel, 5000);
    }
}

// Initialize
initDashboardParallel();

// ============================================
// REFRESH INTERVALS
// ============================================

// LIVE positions: refresh every 30 seconds with retry
setInterval(() => {
    fetchWithRetry('/api/positions', 'positionsRetries', processPositionsData);
}, 30000);

// Trades: refresh every 3 minutes with retry
setInterval(() => {
    fetchWithRetry('/api/closed-trades', 'tradesRetries', processClosedTradesData);
}, 180000);

// Staleness watchdog: check every 15 seconds if data has gone stale
setInterval(updateStalenessCheck, 15000);
