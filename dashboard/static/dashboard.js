// LOX FUND Dashboard v2 - Resume Grade (Streamlined)

let dismissedAlertTimestamp = null;

function dismissAlert() {
    const alert = document.getElementById('regime-alert');
    if (alert) {
        alert.style.display = 'none';
        dismissedAlertTimestamp = new Date().toISOString();
    }
}

function showRegimeAlert(details) {
    const alert = document.getElementById('regime-alert');
    const alertText = document.getElementById('alert-text');
    if (!alert || !alertText || !details || details.length === 0) return;
    
    const change = details[0];
    alertText.textContent = `${change.indicator} shifted: ${change.from} → ${change.to}`;
    alert.style.display = 'flex';
}

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

// ============================================
// POSITIONS DATA (with AUM)
// ============================================
function processPositionsData(data) {
    if (data.error) {
        console.error('Positions API Error:', data.error);
    }
    
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
    if (data.sp500_return !== undefined && data.sp500_return !== null) {
        const sp500 = data.sp500_return;
        const alpha = data.alpha_sp500;
        let benchmarkHtml = `S&P 500: ${formatPercent(sp500)}`;
        if (alpha !== undefined && alpha !== null) {
            const alphaClass = alpha >= 0 ? 'positive' : 'negative';
            benchmarkHtml += ` <span class="alpha ${alphaClass}">${formatPercent(alpha, 1)} alpha</span>`;
        }
        heroBenchmark.innerHTML = benchmarkHtml;
    }
    
    // HERO: AUM and investor count
    const heroAum = document.getElementById('hero-aum');
    if (heroAum && data.aum !== undefined) {
        const investorCount = data.investor_count || 0;
        heroAum.innerHTML = `<span class="aum-badge">${investorCount} investors</span> · <span class="aum-value">${formatCurrency(data.aum)} AUM</span>`;
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
    
    // POSITIONS: P&L badge
    const posPnlEl = document.getElementById('positions-pnl');
    if (data.positions && data.positions.length > 0) {
        let totalPnl = 0;
        let totalCost = 0;
        data.positions.forEach(p => {
            totalPnl += p.pnl || 0;
            totalCost += Math.abs((p.market_value || 0) - (p.pnl || 0));
        });
        const pnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0;
        posPnlEl.textContent = formatPercent(pnlPct, 1);
        posPnlEl.className = 'badge-pnl ' + (pnlPct >= 0 ? 'positive' : 'negative');
    }
    
    // POSITIONS: Render list with thesis
    const positionsList = document.getElementById('positions-list');
    if (!data.positions || data.positions.length === 0) {
        positionsList.innerHTML = '<div class="loading">No open positions</div>';
    } else {
        positionsList.innerHTML = data.positions.map(pos => {
            const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
            
            // Format symbol
            let symbol = pos.symbol;
            let details = '';
            if (pos.opt_info) {
                const type = (pos.opt_info.opt_type || '').toUpperCase().startsWith('C') ? 'C' : 'P';
                symbol = `${pos.opt_info.underlying} $${pos.opt_info.strike}${type}`;
                details = `<span class="position-expiry">${pos.opt_info.expiry}</span>`;
            }
            
            // Thesis (from API or placeholder)
            const thesis = pos.thesis || 'Loading thesis...';
            
            return `
                <div class="position-card">
                    <div class="position-header">
                        <div class="position-symbol-row">
                            <span class="position-symbol">${symbol}</span>
                            ${details}
                        </div>
                        <div class="position-pnl ${pnlClass}">
                            ${formatCurrency(pos.pnl || 0)}
                            <span class="pnl-percent">${formatPercent(pos.pnl_pct || 0, 1)}</span>
                        </div>
                    </div>
                    <div class="position-meta">
                        <span class="position-qty">${pos.qty > 0 ? '+' : ''}${pos.qty} @ ${formatCurrency(Math.abs(pos.market_value || 0) / Math.abs(pos.qty || 1))}</span>
                        <span class="position-value">${formatCurrency(Math.abs(pos.market_value || 0))}</span>
                    </div>
                    <div class="position-thesis">
                        <span class="thesis-label">Thesis:</span>
                        <span class="thesis-text" id="thesis-${pos.symbol.replace(/[^a-zA-Z0-9]/g, '_')}">${thesis}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        // Fetch thesis for positions if not already included
        if (!data.positions[0]?.thesis) {
            fetchPositionThesis(data.positions);
        }
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
// MARKET CONTEXT (Simplified - Regime + Insight)
// ============================================
function processMarketContextData(data) {
    // Regime badge
    const regimeBadge = document.getElementById('regime-badge');
    if (regimeBadge && data.traffic_lights?.regime) {
        const regime = data.traffic_lights.regime;
        regimeBadge.textContent = regime.label || '—';
        regimeBadge.className = 'regime-badge ' + (regime.label || '').toLowerCase().replace(' ', '-');
    }
    
    // AI Insight (1 sentence summary)
    const insightText = document.getElementById('insight-text');
    const insightTime = document.getElementById('insight-time');
    if (insightText) {
        if (data.summary) {
            // Use condensed 1-sentence summary if available
            insightText.textContent = data.summary;
        } else if (data.analysis) {
            // Fallback to first sentence of full analysis
            const firstSentence = data.analysis.split('.')[0] + '.';
            insightText.textContent = firstSentence;
        } else {
            insightText.textContent = 'Market analysis loading...';
        }
        if (insightTime) insightTime.textContent = `Updated ${formatTime()}`;
    }
    
    // Regime change alert
    if (data.regime_changed && data.regime_change_details) {
        if (!dismissedAlertTimestamp || new Date(data.timestamp) > new Date(dismissedAlertTimestamp)) {
            showRegimeAlert(data.regime_change_details);
        }
    }
}

// ============================================
// TRADE PERFORMANCE (Closed Trades)
// ============================================
function processClosedTradesData(data) {
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
    } else {
        if (statsEl) statsEl.textContent = 'NO CLOSED TRADES';
        if (realizedPnlEl) {
            realizedPnlEl.textContent = '$0';
            realizedPnlEl.className = 'badge-pnl';
        }
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

// ============================================
// INIT & REFRESH (Streamlined - 3 endpoints)
// ============================================

async function initDashboardParallel() {
    console.log('[Dashboard] Starting parallel load (v2 streamlined)...');
    const startTime = performance.now();
    
    // Only fetch 3 endpoints now (removed investors, news, domains)
    const fetches = [
        fetch('/api/positions').then(r => r.json()).catch(e => ({ error: e.message, positions: [] })),
        fetch('/api/regime-analysis').then(r => r.json()).catch(e => ({ error: e.message })),
        fetch('/api/closed-trades').then(r => r.json()).catch(e => ({ error: e.message, trades: [] })),
    ];
    
    try {
        const [positionsData, marketData, tradesData] = await Promise.all(fetches);
        
        processPositionsData(positionsData);
        processMarketContextData(marketData);
        processClosedTradesData(tradesData);
        
        const elapsed = (performance.now() - startTime).toFixed(0);
        console.log(`[Dashboard] Load complete in ${elapsed}ms`);
    } catch (err) {
        console.error('[Dashboard] Load error:', err);
    }
}

// Initialize
initDashboardParallel();

// ============================================
// REFRESH INTERVALS
// ============================================

// LIVE data: Positions refresh every 10 seconds
setInterval(() => {
    fetch('/api/positions')
        .then(r => r.json())
        .then(processPositionsData)
        .catch(console.error);
}, 10000);

// Market context & trades: Refresh every 5 minutes
setInterval(() => {
    Promise.all([
        fetch('/api/regime-analysis').then(r => r.json()).then(processMarketContextData).catch(console.error),
        fetch('/api/closed-trades').then(r => r.json()).then(processClosedTradesData).catch(console.error),
    ]);
}, 300000);
