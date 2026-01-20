// LOX FUND Dashboard - Auto-refresh functionality

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value);
}

function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function formatSymbol(symbol, optInfo) {
    if (optInfo) {
        // Handle both 'C'/'P' and 'CALL'/'PUT' formats
        let type = 'P';
        if (optInfo.opt_type === 'C' || optInfo.opt_type === 'CALL' || optInfo.opt_type === 'call') {
            type = 'C';
        } else if (optInfo.opt_type === 'P' || optInfo.opt_type === 'PUT' || optInfo.opt_type === 'put') {
            type = 'P';
        }
        return `${optInfo.underlying} ${optInfo.strike}${type} ${optInfo.expiry}`;
    }
    return symbol;
}

function updateDashboard() {
    fetch('/api/positions')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Update summary - always show values even if there's an error
            const navEl = document.getElementById('nav-value');
            if (data.nav_equity && data.nav_equity > 0) {
                navEl.textContent = formatCurrency(data.nav_equity);
            } else if (data.error) {
                navEl.textContent = 'Error';
            } else {
                navEl.textContent = formatCurrency(0);
            }
            navEl.className = 'summary-value';

            const pnlEl = document.getElementById('total-pnl');
            pnlEl.textContent = formatCurrency(data.total_pnl || 0);
            pnlEl.className = 'summary-value ' + ((data.total_pnl || 0) >= 0 ? 'positive' : 'negative');

            if (data.error) {
                console.error('Error:', data.error);
                document.getElementById('positions-body').innerHTML = 
                    `<tr><td colspan="5" class="loading">Error loading positions: ${data.error}</td></tr>`;
                return;
            }

            const updateEl = document.getElementById('last-update');
            updateEl.textContent = formatTimestamp(data.timestamp);
            updateEl.className = 'summary-value';

            // Update macro indicators
            const macroContainer = document.getElementById('macro-indicators');
            if (data.macro_indicators && data.macro_indicators.length > 0) {
                macroContainer.innerHTML = data.macro_indicators.map(ind => {
                    const valueStr = ind.unit === 'bps' 
                        ? `${ind.value.toFixed(0)} ${ind.unit}`
                        : ind.unit === '%'
                        ? `${ind.value.toFixed(2)}${ind.unit}`
                        : `${ind.value.toFixed(1)}`;
                    
                    const inRange = ind.in_range === true;
                    const itemClass = inRange ? 'macro-item in-range' : 'macro-item';
                    const contextClass = inRange ? 'macro-context in-range' : 'macro-context';
                    const targetText = ind.target || '';
                    const contextText = ind.context || '';
                    
                    const description = ind.description ? `<div class="macro-description">${ind.description}</div>` : '';
                    return `
                        <div class="${itemClass}">
                            <div class="macro-label">${ind.label}</div>
                            ${description}
                            <div class="macro-value">${valueStr}</div>
                            ${targetText ? `<div class="macro-target">Target: <span class="target-label">${targetText}</span></div>` : ''}
                            ${contextText ? `<div class="${contextClass}">${contextText}</div>` : ''}
                            <div class="macro-asof">${ind.asof}</div>
                        </div>
                    `;
                }).join('');
            } else {
                macroContainer.innerHTML = '<div class="macro-item"><div class="macro-label">No data</div><div class="macro-value">—</div></div>';
            }

            // Update positions table
            const tbody = document.getElementById('positions-body');
            if (data.positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="loading">No positions</td></tr>';
                return;
            }

            tbody.innerHTML = data.positions.map(pos => {
                const pnlClass = pos.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                let optDetails = '';
                if (pos.opt_info) {
                    // Normalize opt_type to C or P for display
                    let type = 'P';
                    if (pos.opt_info.opt_type === 'C' || pos.opt_info.opt_type === 'CALL' || pos.opt_info.opt_type === 'call') {
                        type = 'C';
                    } else if (pos.opt_info.opt_type === 'P' || pos.opt_info.opt_type === 'PUT' || pos.opt_info.opt_type === 'put') {
                        type = 'P';
                    }
                    optDetails = `<div class="opt-details">${pos.opt_info.expiry} | Strike ${pos.opt_info.strike}${type}</div>`;
                }
                
                return `
                    <tr>
                        <td>
                            <div class="symbol">${formatSymbol(pos.symbol, pos.opt_info)}</div>
                            ${optDetails}
                        </td>
                        <td>${pos.qty > 0 ? '+' : ''}${pos.qty.toFixed(0)}</td>
                        <td>${formatCurrency(Math.abs(pos.market_value))}</td>
                        <td class="${pnlClass}">${formatCurrency(pos.pnl)}</td>
                        <td class="${pnlClass}">${formatPercent(pos.pnl_pct)}</td>
                    </tr>
                `;
            }).join('');
        })
        .catch(error => {
            console.error('Fetch error:', error);
            document.getElementById('nav-value').textContent = 'Error';
            document.getElementById('total-pnl').textContent = 'Error';
            document.getElementById('positions-body').innerHTML = 
                '<tr><td colspan="5" class="loading">Error connecting to server. Please refresh.</td></tr>';
        });
}

// Format markdown-style text to HTML
function formatAnalysis(text) {
    if (!text) return '';
    
    // Convert markdown-style formatting to HTML
    let html = text
        // Bold text with **
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Headings with ** at start
        .replace(/^(\*\*[^*]+\*\*)$/gm, '<h3>$1</h3>')
        // Lists
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^• (.+)$/gm, '<li>$1</li>')
        // Wrap consecutive li in ul
        .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
        // Paragraphs
        .replace(/\n\n/g, '</p><p>')
        // Line breaks
        .replace(/\n/g, '<br>');
    
    // Wrap in paragraph if not already
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }
    
    return html;
}

// Fetch and display Palmer's cached regime analysis
function fetchRegimeAnalysis() {
    const container = document.getElementById('regime-analysis');
    const timestamp = document.getElementById('analysis-timestamp');
    
    fetch('/api/regime-analysis')
        .then(response => response.json())
        .then(data => {
            if (data.error && !data.analysis) {
                container.innerHTML = `<div class="analysis-error">Error: ${data.error}</div>`;
                timestamp.textContent = '';
                return;
            }
            
            if (data.analysis) {
                container.innerHTML = formatAnalysis(data.analysis);
                
                // Show when analysis was generated and next refresh time
                let timestampText = '';
                if (data.timestamp) {
                    timestampText = `Analysis generated: ${formatTimestamp(data.timestamp)}`;
                }
                if (data.next_refresh) {
                    const nextRefresh = new Date(data.next_refresh);
                    const now = new Date();
                    const minsUntilRefresh = Math.max(0, Math.round((nextRefresh - now) / 60000));
                    timestampText += ` • Next refresh in ${minsUntilRefresh} min`;
                }
                timestamp.textContent = timestampText;
            } else {
                // Analysis not yet generated - show loading
                container.innerHTML = `
                    <div class="analysis-loading">
                        <div class="spinner"></div>
                        <span>Palmer is generating initial analysis...</span>
                    </div>
                `;
                timestamp.textContent = '';
                // Retry in 10 seconds
                setTimeout(fetchRegimeAnalysis, 10000);
            }
        })
        .catch(error => {
            console.error('Analysis fetch error:', error);
            container.innerHTML = '<div class="analysis-error">Unable to load Palmer analysis. Server may be starting up.</div>';
            timestamp.textContent = '';
            // Retry in 10 seconds
            setTimeout(fetchRegimeAnalysis, 10000);
        });
}

// Initial load
updateDashboard();
fetchRegimeAnalysis();

// Auto-refresh positions every 5 minutes
setInterval(updateDashboard, 300000);

// Check for new Palmer analysis every 5 minutes (server refreshes every 30 min)
setInterval(fetchRegimeAnalysis, 300000);
