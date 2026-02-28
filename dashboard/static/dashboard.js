// LOX FUND Dashboard v3 - Reliable & Dynamic

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
        // Still process whatever data came back (may be stale cache fallback)
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

    // HERO: Benchmark comparison
    const heroBenchmark = document.getElementById('hero-benchmark');
    let benchParts = [];
    if (data.sp500_return !== undefined && data.sp500_return !== null) {
        benchParts.push(`<span class="bench-item">S&P 500: ${formatPercent(data.sp500_return)}</span>`);
    }
    if (data.macro_hf_return !== undefined && data.macro_hf_return !== null) {
        benchParts.push(`<span class="bench-item">Macro Hedge Fund Index: ${formatPercent(data.macro_hf_return)}</span>`);
    }
    if (benchParts.length) {
        heroBenchmark.innerHTML = benchParts.join('<span class="bench-sep">·</span>');
    }

    // HERO: AUM and investor count
    const heroAum = document.getElementById('hero-aum');
    if (heroAum && data.aum !== undefined) {
        const investorCount = data.investor_count || 0;
        heroAum.innerHTML = `<span class="aum-badge">${investorCount} investors</span> · <span class="aum-value">${formatCurrency(data.aum)} seed capital</span>`;
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
            const safeId = pos.symbol.replace(/[^a-zA-Z0-9]/g, '_');

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

            // Thesis & indicators for expandable detail
            const thesis = pos.thesis || '';
            const indicators = pos.indicators || [];
            const hasDetail = thesis || indicators.length > 0;

            let detailHtml = '';
            if (hasDetail) {
                let indRows = '';
                if (indicators.length > 0) {
                    indRows = `
                        <table class="pos-detail-indicators">
                            <thead><tr><th>Indicator</th><th>Current</th><th>Target</th><th>Invalidation</th><th></th></tr></thead>
                            <tbody>${indicators.map(ind => `
                                <tr>
                                    <td>${escapeHtml(ind.name || '')}</td>
                                    <td>${escapeHtml(ind.current_value || '—')}</td>
                                    <td>${escapeHtml(ind.target_value || '—')}</td>
                                    <td>${escapeHtml(ind.invalidation_value || '—')}</td>
                                    <td><span class="ind-dot ind-dot--${ind.status || 'neutral'}"></span></td>
                                </tr>`).join('')}
                            </tbody>
                        </table>`;
                }
                detailHtml = `
                    <tr class="pos-detail-row" id="detail-${safeId}" style="display:none;">
                        <td colspan="7">
                            <div class="pos-detail-content">
                                ${thesis ? `<div class="pos-detail-thesis" id="thesis-${safeId}">${escapeHtml(thesis)}</div>` : ''}
                                ${indRows}
                            </div>
                        </td>
                    </tr>`;
            }

            return `
                <tr class="pos-row${hasDetail ? ' pos-row--expandable' : ''}" data-detail="detail-${safeId}">
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
                </tr>
                ${detailHtml}`;
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
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;

        // Wire up expandable rows
        positionsList.querySelectorAll('.pos-row--expandable').forEach(row => {
            row.addEventListener('click', () => {
                const detailId = row.getAttribute('data-detail');
                const detail = document.getElementById(detailId);
                if (!detail) return;
                const isOpen = detail.style.display !== 'none';
                detail.style.display = isOpen ? 'none' : 'table-row';
                row.classList.toggle('pos-row--open', !isOpen);
            });
        });

        // Lazily fetch AI-generated thesis
        fetchPositionThesis(data.positions);
    }

    // Update footer
    document.getElementById('footer-time').textContent = `Updated ${formatTime()}`;
}


// ============================================
// POSITION THESIS (AI Generated)
// ============================================
async function fetchPositionThesis(positions) {
    try {
        const response = await fetch('/api/position-thesis');
        if (!response.ok) return;
        const data = await response.json();

        if (data.error || !data.theses) return;

        // Update each position's thesis
        Object.entries(data.theses).forEach(([symbol, thesis]) => {
            const safeId = symbol.replace(/[^a-zA-Z0-9]/g, '_');
            const el = document.getElementById(`thesis-${safeId}`);
            if (el) {
                el.textContent = thesis;
            }
        });
    } catch (err) {
        console.error('Thesis fetch error:', err);
    }
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
// INIT & REFRESH
// ============================================

async function initDashboardParallel() {
    console.log('[Dashboard] Starting parallel load...');
    const startTime = performance.now();

    const fetches = [
        fetch('/api/positions').then(r => r.json()).catch(e => ({ error: e.message, positions: [] })),
        fetch('/api/closed-trades').then(r => r.json()).catch(e => ({ error: e.message, trades: [] })),
        fetch('/api/position-thesis').then(r => r.json()).catch(e => ({ error: e.message })),
    ];

    try {
        const [positionsData, tradesData, thesisData] = await Promise.all(fetches);

        processPositionsData(positionsData);
        processClosedTradesData(tradesData);

        if (thesisData && thesisData.theses) {
            Object.entries(thesisData.theses).forEach(([symbol, thesis]) => {
                const safeId = symbol.replace(/[^a-zA-Z0-9]/g, '_');
                const el = document.getElementById(`thesis-${safeId}`);
                if (el) el.textContent = thesis;
            });
        }

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
