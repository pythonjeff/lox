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
        
        // Lazily fetch AI-generated thesis (upgrades the simple fallback thesis)
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
// MARKET CONTEXT (Enhanced - Regime + Metrics + Insight)
// ============================================
function processMarketContextData(data) {
    // Regime badge
    const regimeBadge = document.getElementById('regime-badge');
    let regimeLabel = '—';
    if (regimeBadge && data.traffic_lights?.regime) {
        const regime = data.traffic_lights.regime;
        regimeLabel = regime.label || '—';
        regimeBadge.textContent = regimeLabel;
        regimeBadge.className = 'regime-badge ' + (regimeLabel).toLowerCase().replace(' ', '-');
    }
    
    // Key metrics with context
    if (data.traffic_lights) {
        const tl = data.traffic_lights;
        
        // VIX - Fear Gauge
        const vixEl = document.getElementById('vix-value');
        const vixContext = document.getElementById('vix-context');
        if (vixEl && tl.volatility) {
            const vixStr = tl.volatility.value || '—';
            // Extract just the number for display
            const vixNum = parseFloat(vixStr.replace(/[^0-9.]/g, ''));
            vixEl.textContent = vixNum ? vixNum.toFixed(1) : '—';
            
            // Color and context based on level
            if (vixNum < 15) {
                vixEl.className = 'regime-metric-value green';
                if (vixContext) vixContext.textContent = 'Complacent (<15)';
            } else if (vixNum < 20) {
                vixEl.className = 'regime-metric-value green';
                if (vixContext) vixContext.textContent = 'Calm (15-20)';
            } else if (vixNum < 25) {
                vixEl.className = 'regime-metric-value yellow';
                if (vixContext) vixContext.textContent = 'Elevated (20-25)';
            } else if (vixNum < 30) {
                vixEl.className = 'regime-metric-value yellow';
                if (vixContext) vixContext.textContent = 'Fear (25-30)';
            } else {
                vixEl.className = 'regime-metric-value red';
                if (vixContext) vixContext.textContent = 'Panic (>30)';
            }
        }
        
        // HY Spread - Credit Stress (basis points)
        const creditEl = document.getElementById('credit-value');
        const creditContext = document.getElementById('credit-context');
        if (creditEl && tl.credit) {
            const creditStr = tl.credit.value || '—';
            const creditNum = parseFloat(creditStr.replace(/[^0-9.]/g, ''));
            creditEl.textContent = creditNum ? `${Math.round(creditNum)}bp` : '—';
            
            // Color and context based on level
            if (creditNum < 300) {
                creditEl.className = 'regime-metric-value green';
                if (creditContext) creditContext.textContent = 'Tight (<300bp)';
            } else if (creditNum < 400) {
                creditEl.className = 'regime-metric-value yellow';
                if (creditContext) creditContext.textContent = 'Normal (300-400bp)';
            } else if (creditNum < 500) {
                creditEl.className = 'regime-metric-value yellow';
                if (creditContext) creditContext.textContent = 'Wide (400-500bp)';
            } else {
                creditEl.className = 'regime-metric-value red';
                if (creditContext) creditContext.textContent = 'Stress (>500bp)';
            }
        }
        
        // 10Y Yield - Rate Pressure
        const ratesEl = document.getElementById('rates-value');
        const ratesContext = document.getElementById('rates-context');
        if (ratesEl && tl.rates) {
            const ratesStr = tl.rates.value || '—';
            const ratesNum = parseFloat(ratesStr.replace(/[^0-9.]/g, ''));
            ratesEl.textContent = ratesNum ? `${ratesNum.toFixed(2)}%` : '—';
            
            // Color and context based on level
            if (ratesNum < 3.5) {
                ratesEl.className = 'regime-metric-value green';
                if (ratesContext) ratesContext.textContent = 'Accommodative (<3.5%)';
            } else if (ratesNum < 4.25) {
                ratesEl.className = 'regime-metric-value yellow';
                if (ratesContext) ratesContext.textContent = 'Neutral (3.5-4.25%)';
            } else if (ratesNum < 4.75) {
                ratesEl.className = 'regime-metric-value yellow';
                if (ratesContext) ratesContext.textContent = 'Tight (4.25-4.75%)';
            } else {
                ratesEl.className = 'regime-metric-value red';
                if (ratesContext) ratesContext.textContent = 'Restrictive (>4.75%)';
            }
        }
        
        // CPI YoY - Inflation
        const cpiEl = document.getElementById('cpi-value');
        const cpiContext = document.getElementById('cpi-context');
        if (cpiEl && tl.inflation) {
            const cpiStr = tl.inflation.value || '—';
            const cpiNum = parseFloat(cpiStr.replace(/[^0-9.]/g, ''));
            cpiEl.textContent = cpiNum ? `${cpiNum.toFixed(1)}%` : '—';
            
            // Color and context based on level
            if (cpiNum > 4.0) {
                cpiEl.className = 'regime-metric-value red';
                if (cpiContext) cpiContext.textContent = 'Hot (>4%)';
            } else if (cpiNum > 3.0) {
                cpiEl.className = 'regime-metric-value yellow';
                if (cpiContext) cpiContext.textContent = 'Sticky (3-4%)';
            } else if (cpiNum > 2.0) {
                cpiEl.className = 'regime-metric-value green';
                if (cpiContext) cpiContext.textContent = 'Target (2-3%)';
            } else {
                cpiEl.className = 'regime-metric-value green';
                if (cpiContext) cpiContext.textContent = 'Cool (<2%)';
            }
        }
        
        // 2s10s Yield Curve Spread
        const curveEl = document.getElementById('curve-value');
        const curveContext = document.getElementById('curve-context');
        if (curveEl && tl.yield_curve) {
            const curveStr = tl.yield_curve.value || '—';
            const curveNum = parseFloat(curveStr.replace(/[^0-9.-]/g, ''));
            curveEl.textContent = !isNaN(curveNum) ? `${curveNum.toFixed(0)}bp` : '—';
            
            // Color and context based on level
            if (curveNum < -50) {
                curveEl.className = 'regime-metric-value red';
                if (curveContext) curveContext.textContent = 'Deep inversion';
            } else if (curveNum < 0) {
                curveEl.className = 'regime-metric-value yellow';
                if (curveContext) curveContext.textContent = 'Inverted';
            } else if (curveNum < 50) {
                curveEl.className = 'regime-metric-value yellow';
                if (curveContext) curveContext.textContent = 'Flat (0-50bp)';
            } else {
                curveEl.className = 'regime-metric-value green';
                if (curveContext) curveContext.textContent = 'Steep (>50bp)';
            }
        }
    }
    
    // AI Insight
    const insightText = document.getElementById('insight-text');
    const insightTime = document.getElementById('insight-time');
    const implicationText = document.getElementById('implication-text');
    
    if (insightText) {
        if (data.summary) {
            insightText.textContent = data.summary;
        } else if (data.analysis) {
            const firstSentence = data.analysis.split('.')[0] + '.';
            insightText.textContent = firstSentence;
        } else {
            insightText.textContent = 'Market analysis loading...';
        }
        if (insightTime) insightTime.textContent = formatTime();
    }
    
    // Portfolio implication based on regime (mean-reversion / bearish bias strategy)
    if (implicationText) {
        const implications = {
            'RISK-ON': 'Complacency rising—prime environment for mean reversion puts on extended names. Calls serve as hedges against short delta. Low VIX = cheap vol for long-dated protection.',
            'CAUTIOUS': 'Inflection point. Tighten stops on existing puts, scale into hedges. Watch for breakdown confirmation before adding bearish exposure.',
            'RISK-OFF': 'Thesis playing out—puts gaining. Consider taking profits on winners, let hedges (calls) decay. Elevated vol = expensive to initiate new positions.',
        };
        implicationText.textContent = implications[regimeLabel] || 'Monitoring for mean reversion setups.';
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
    
    // ── Operating P&L (30-day net after costs) ──
    renderOperatingPnl(data.trades || []);
}

function renderOperatingPnl(trades) {
    const section = document.getElementById('ops-pnl-section');
    if (!section) return;
    
    // Monthly infrastructure costs
    const MONTHLY_COSTS = [
        { name: 'Alpaca', amount: 99 },
        { name: 'FMP', amount: 69 },
    ];
    const totalCosts = MONTHLY_COSTS.reduce((sum, c) => sum + c.amount, 0);
    
    // Calculate realized P&L from trades closed in the last 30 days
    const now = new Date();
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    
    let pnl30d = 0;
    let tradeCount30d = 0;
    
    for (const t of trades) {
        // exit_date may come as RFC 2822 ("Thu, 05 Feb 2026 15:54:21 GMT") or ISO string
        let exitDate = null;
        if (t.exit_date) {
            exitDate = new Date(t.exit_date);
            if (isNaN(exitDate.getTime())) exitDate = null;
        }
        if (exitDate && exitDate >= thirtyDaysAgo) {
            pnl30d += t.pnl || 0;
            tradeCount30d++;
        }
    }
    
    const netPnl = pnl30d - totalCosts;
    
    section.style.display = 'block';
    
    // Period label
    const periodEl = document.getElementById('ops-pnl-period');
    if (periodEl) {
        periodEl.textContent = `${tradeCount30d} trades closed in last 30d`;
    }
    
    // Gross P&L
    const grossEl = document.getElementById('ops-gross-pnl');
    if (grossEl) {
        grossEl.textContent = formatCurrency(pnl30d);
        grossEl.className = 'ops-pnl-value ' + (pnl30d >= 0 ? 'positive' : 'negative');
    }
    
    // Costs
    const costsEl = document.getElementById('ops-costs');
    if (costsEl) {
        costsEl.textContent = formatCurrency(-totalCosts);
    }
    
    // Cost breakdown
    const breakdownEl = document.getElementById('ops-breakdown');
    if (breakdownEl) {
        breakdownEl.textContent = MONTHLY_COSTS.map(c => `${c.name} $${c.amount}`).join(' · ');
    }
    
    // Net P&L
    const netEl = document.getElementById('ops-net-pnl');
    if (netEl) {
        netEl.textContent = formatCurrency(netPnl);
        netEl.className = 'ops-pnl-value ' + (netPnl >= 0 ? 'positive' : 'negative');
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
// REGIME PERFORMANCE (Resume Builder - Features 1-6)
// ============================================

let regimeEquityChart = null;

const REGIME_COLORS = {
    'RISK-ON': { bg: 'rgba(0, 168, 107, 0.18)', border: '#00a86b', text: '#00a86b', badge: 'risk-on' },
    'CAUTIOUS': { bg: 'rgba(245, 166, 35, 0.22)', border: '#f5a623', text: '#f5a623', badge: 'cautious' },
    'RISK-OFF': { bg: 'rgba(230, 57, 70, 0.18)', border: '#e63946', text: '#e63946', badge: 'risk-off' },
};

function processRegimePerformance(data) {
    if (data.error && !data.by_regime) {
        console.error('[Regime Perf] Error:', data.error);
        return;
    }

    // ── Edge Summary Card ──
    const edgeCard = document.getElementById('edge-card');
    if (edgeCard && data.edge) {
        const e = data.edge;
        edgeCard.style.display = 'block';

        // Total P&L
        const pnlEl = document.getElementById('edge-total-pnl');
        if (pnlEl && e.total_pnl !== undefined) {
            pnlEl.textContent = formatCurrency(e.total_pnl);
            pnlEl.className = 'edge-stat-value ' + (e.total_pnl >= 0 ? 'positive' : 'negative');
        }

        // Best environment
        const bestEl = document.getElementById('edge-best-regime');
        const bestDetail = document.getElementById('edge-best-detail');
        if (bestEl && e.best_regime) {
            bestEl.textContent = e.best_regime;
            bestEl.className = 'edge-stat-value ' + (REGIME_COLORS[e.best_regime]?.badge || '');
            if (bestDetail) {
                let detail = 'Best Environment';
                if (e.best_regime_wr) detail += ` (${e.best_regime_wr}% WR)`;
                bestDetail.textContent = detail;
            }
        }

        // Win rate
        const wrEl = document.getElementById('edge-win-rate');
        if (wrEl && e.overall_win_rate !== undefined) {
            wrEl.textContent = e.overall_win_rate + '%';
            wrEl.className = 'edge-stat-value ' + (e.overall_win_rate >= 55 ? 'positive' : '');
        }

        // Profit factor
        const pfEl = document.getElementById('edge-profit-factor');
        if (pfEl && e.profit_factor !== undefined) {
            const pf = e.profit_factor;
            pfEl.textContent = pf >= 999 ? '∞' : pf.toFixed(1) + 'x';
            pfEl.className = 'edge-stat-value ' + (pf >= 1.5 ? 'positive' : pf >= 1.0 ? '' : 'negative');
        }
    }

    // ── Performance-by-Regime Table (enriched with hold time & puts/calls) ──
    const perfSection = document.getElementById('regime-perf-section');
    const perfBody = document.getElementById('regime-perf-body');
    if (perfBody && data.by_regime && Object.keys(data.by_regime).length > 0) {
        perfSection.style.display = 'block';
        const order = ['RISK-ON', 'CAUTIOUS', 'RISK-OFF'];
        const sorted = Object.entries(data.by_regime).sort(
            (a, b) => order.indexOf(a[0]) - order.indexOf(b[0])
        );
        perfBody.innerHTML = sorted.map(([regime, s]) => {
            const rc = REGIME_COLORS[regime] || {};
            const pnlClass = s.total_pnl >= 0 ? 'positive' : 'negative';
            const hold = s.avg_hold_days !== null ? s.avg_hold_days + 'd' : '—';
            const pc = `${s.puts || 0}P/${s.calls || 0}C`;
            return `<tr>
                <td><span class="regime-badge ${rc.badge || ''}">${regime}</span></td>
                <td>${s.trades}</td>
                <td>${s.win_rate.toFixed(0)}%</td>
                <td class="${s.avg_return >= 0 ? 'positive' : 'negative'}">${s.avg_return >= 0 ? '+' : ''}${s.avg_return.toFixed(1)}%</td>
                <td>${hold}</td>
                <td>${pc}</td>
                <td class="${pnlClass}">${formatCurrency(s.total_pnl)}</td>
            </tr>`;
        }).join('');
    }

    // ── Regime-Aware Equity Curve (from closed trades) ──
    const chartSection = document.getElementById('regime-chart-section');
    if (chartSection && data.equity_series && data.equity_series.length > 1 && typeof Chart !== 'undefined') {
        chartSection.style.display = 'block';
        renderRegimeEquityCurve(data.equity_series, data.regime_bands);
    }

}

function renderRegimeEquityCurve(equitySeries, regimeBands) {
    const canvas = document.getElementById('regime-equity-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Destroy previous chart
    if (regimeEquityChart) {
        regimeEquityChart.destroy();
        regimeEquityChart = null;
    }

    // Build data as {x, y} for time-scale axis
    const chartData = equitySeries.map(p => ({ x: p.date, y: p.pnl }));

    // Build regime background bands as Chart.js annotations
    const bandAnnotations = {};
    if (regimeBands && regimeBands.length > 0) {
        regimeBands.forEach((band, i) => {
            const rc = REGIME_COLORS[band.regime] || {};
            bandAnnotations['band' + i] = {
                type: 'box',
                xMin: band.start,
                xMax: band.end,
                backgroundColor: rc.bg || 'rgba(200,200,200,0.1)',
                borderWidth: 0,
                drawTime: 'beforeDatasetsDraw',
            };
        });
    }

    regimeEquityChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Cumulative P&L',
                data: chartData,
                borderColor: '#0066ff',
                backgroundColor: 'rgba(0, 102, 255, 0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                annotation: {
                    annotations: bandAnnotations,
                },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const val = ctx.parsed.y;
                            return `P&L: ${val >= 0 ? '+' : ''}$${val.toFixed(0)}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: { day: 'MMM d' },
                        tooltipFormat: 'MMM d, yyyy',
                    },
                    display: true,
                    grid: { display: false },
                    ticks: {
                        font: { size: 10, family: 'Inter' },
                        color: '#999',
                        maxTicksLimit: 8,
                    }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    ticks: {
                        font: { size: 10, family: 'Inter' },
                        color: '#999',
                        callback: function(val) { return (val >= 0 ? '+' : '') + '$' + val.toFixed(0); }
                    }
                }
            }
        }
    });

    // Build legend for regime bands
    const legendEl = document.getElementById('regime-chart-legend');
    if (legendEl) {
        legendEl.innerHTML = ['RISK-ON', 'CAUTIOUS', 'RISK-OFF'].map(r => {
            const rc = REGIME_COLORS[r];
            return `<span class="chart-legend-item">
                <span class="chart-legend-dot" style="background:${rc.border}"></span>${r}
            </span>`;
        }).join('');
    }
}

// ============================================
// INIT & REFRESH (4 endpoints)
// ============================================

async function initDashboardParallel() {
    console.log('[Dashboard] Starting parallel load (v3 regime resume)...');
    const startTime = performance.now();
    
    const fetches = [
        fetch('/api/positions').then(r => r.json()).catch(e => ({ error: e.message, positions: [] })),
        fetch('/api/regime-analysis').then(r => r.json()).catch(e => ({ error: e.message })),
        fetch('/api/closed-trades').then(r => r.json()).catch(e => ({ error: e.message, trades: [] })),
        fetch('/api/regime-performance').then(r => r.json()).catch(e => ({ error: e.message, by_regime: {} })),
    ];
    
    try {
        const [positionsData, marketData, tradesData, regimePerf] = await Promise.all(fetches);
        
        processPositionsData(positionsData);
        processMarketContextData(marketData);
        processClosedTradesData(tradesData);
        processRegimePerformance(regimePerf);
        
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

// LIVE data: Positions refresh every 30 seconds (matches server cache TTL)
setInterval(() => {
    fetch('/api/positions')
        .then(r => r.json())
        .then(processPositionsData)
        .catch(console.error);
}, 30000);

// Market context & trades: Refresh every 5 minutes
setInterval(() => {
    Promise.all([
        fetch('/api/regime-analysis').then(r => r.json()).then(processMarketContextData).catch(console.error),
        fetch('/api/closed-trades').then(r => r.json()).then(processClosedTradesData).catch(console.error),
        fetch('/api/regime-performance').then(r => r.json()).then(processRegimePerformance).catch(console.error),
    ]);
}, 300000);
