/**
 * Lived Inflation Index — Frontend Logic
 *
 * Fetches data from /api/lii/* endpoints, renders Chart.js charts,
 * sortable category table, and scenario profile switching.
 */
(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────────────
    let currentProfile = 'default';
    let timeseriesChart = null;
    let spreadChart = null;
    let weightChart = null;
    let sentimentData = null;
    let showCumulative = false;
    let showSentiment = false;
    let cumulativeData = null;
    let timeseriesRawData = null;

    // Debt overlay state
    let debtOverlay = false;
    let debtCategories = { student: true, credit: true, auto: true };

    // ── Color palette ─────────────────────────────────────────────────
    const COLORS = {
        lii: '#3b82f6',      // blue — LII (matches dashboard accent)
        cpi: '#94a3b8',      // slate gray — CPI (official, muted)
        spread: '#10b981',   // green
        spreadNeg: '#ef4444',// red
        fed: '#64748b',      // gray dashed
        sentiment: '#a78bfa',// violet
        debt: '#f97316',     // hot coral / orange — LII + Debt
        gridLine: '#e5e7eb',
        text: '#6b7280',
    };

    // ── Utilities ─────────────────────────────────────────────────────
    function fmt(n, decimals = 2) {
        if (n === null || n === undefined) return '—';
        return Number(n).toFixed(decimals);
    }

    function fmtDelta(n) {
        if (n === null || n === undefined) return '';
        const sign = n >= 0 ? '+' : '';
        return sign + Number(n).toFixed(2) + ' MoM';
    }

    function spreadColor(val) {
        if (val === null || val === undefined) return '';
        const bps = Math.abs(val * 100);
        if (bps < 50) return 'lii-spread-green';
        if (bps < 150) return 'lii-spread-amber';
        return 'lii-spread-red';
    }

    // ── API Fetchers ──────────────────────────────────────────────────
    async function fetchJSON(url) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Fetch ${url}: ${resp.status}`);
        return resp.json();
    }

    function debtParams() {
        if (!debtOverlay) return '';
        const cats = Object.keys(debtCategories).filter(k => debtCategories[k]);
        return `&debt_overlay=true&debt_categories=${cats.join(',')}`;
    }

    async function fetchCurrent() {
        return fetchJSON(`/api/lii/current?profile=${currentProfile}${debtParams()}`);
    }

    async function fetchTimeseries() {
        return fetchJSON(`/api/lii/timeseries?profile=${currentProfile}${debtParams()}`);
    }

    async function fetchCategories() {
        return fetchJSON(`/api/lii/categories?profile=${currentProfile}${debtParams()}`);
    }

    async function fetchCumulative() {
        return fetchJSON(`/api/lii/cumulative?profile=${currentProfile}`);
    }

    async function fetchSentiment() {
        if (sentimentData) return sentimentData;
        const d = await fetchJSON('/api/lii/sentiment');
        sentimentData = d.data || [];
        return sentimentData;
    }

    async function fetchDebtCurrent() {
        return fetchJSON('/api/debt/current');
    }

    // ── Hero Metrics ──────────────────────────────────────────────────
    function renderHero(data) {
        const liiEl = document.getElementById('lii-value');
        const cpiEl = document.getElementById('cpi-value');
        const spreadEl = document.getElementById('spread-value');
        const liiDelta = document.getElementById('lii-delta');
        const cpiDelta = document.getElementById('cpi-delta');
        const spreadDelta = document.getElementById('spread-delta');
        const dataThrough = document.getElementById('lii-data-through');

        if (liiEl) liiEl.textContent = fmt(data.lii) + '%';
        if (cpiEl) cpiEl.textContent = fmt(data.cpi) + '%';
        if (spreadEl) {
            spreadEl.textContent = fmt(data.spread) + '%';
            const card = spreadEl.closest('.lii-hero-spread');
            if (card) {
                card.classList.remove('lii-spread-green', 'lii-spread-amber', 'lii-spread-red');
                card.classList.add(spreadColor(data.spread));
            }
        }
        if (liiDelta) liiDelta.textContent = fmtDelta(data.lii_mom);
        if (cpiDelta) cpiDelta.textContent = fmtDelta(data.cpi_mom);
        if (spreadDelta) spreadDelta.textContent = fmtDelta(data.spread_mom);
        if (dataThrough) dataThrough.textContent = data.data_month ? `Data through ${data.data_month}` : '';

        // 4th hero card — LII + Debt
        const debtCard = document.getElementById('lii-hero-debt');
        const debtVal = document.getElementById('lii-debt-value');
        const debtDelta = document.getElementById('lii-debt-delta');
        if (debtCard) {
            if (debtOverlay && data.lii_debt != null) {
                debtCard.style.display = '';
                if (debtVal) debtVal.textContent = fmt(data.lii_debt) + '%';
                if (debtDelta) {
                    const bps = Math.round((data.lii_debt - data.lii) * 100);
                    const sign = bps >= 0 ? '+' : '';
                    debtDelta.textContent = `${sign}${bps} bps vs LII`;
                }
            } else {
                debtCard.style.display = 'none';
            }
        }
    }

    // ── Time Series Chart ─────────────────────────────────────────────
    function buildTimeseriesChart(tsData) {
        const ctx = document.getElementById('lii-timeseries-chart');
        if (!ctx) return;

        if (timeseriesChart) timeseriesChart.destroy();

        const labels = tsData.map(d => d.date);
        const liiVals = tsData.map(d => d.lii);
        const cpiVals = tsData.map(d => d.cpi);

        const datasets = [
            {
                label: 'LII (YoY %)',
                data: liiVals,
                borderColor: COLORS.lii,
                backgroundColor: COLORS.lii + '20',
                borderWidth: 2.5,
                pointRadius: 0,
                pointHitRadius: 8,
                tension: 0.3,
                fill: false,
            },
            {
                label: 'CPI-U (YoY %)',
                data: cpiVals,
                borderColor: COLORS.cpi,
                backgroundColor: COLORS.cpi + '20',
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 8,
                tension: 0.3,
                fill: '-1',
            },
        ];

        // 3rd line: LII + Debt (when debt overlay is active)
        if (debtOverlay && tsData.length > 0 && tsData[0].lii_debt != null) {
            datasets.push({
                label: 'LII + Debt (YoY %)',
                data: tsData.map(d => d.lii_debt),
                borderColor: COLORS.debt,
                backgroundColor: COLORS.debt + '15',
                borderWidth: 2.5,
                borderDash: [6, 3],
                pointRadius: 0,
                pointHitRadius: 8,
                tension: 0.3,
                fill: false,
            });
        }

        timeseriesChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: COLORS.text, usePointStyle: true } },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#374151',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
                        },
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'quarter', tooltipFormat: 'MMM yyyy' },
                        grid: { color: COLORS.gridLine },
                        ticks: { color: COLORS.text, maxTicksLimit: 12 },
                    },
                    y: {
                        grid: { color: COLORS.gridLine },
                        ticks: {
                            color: COLORS.text,
                            callback: v => v.toFixed(1) + '%',
                        },
                    },
                },
            },
        });

        return timeseriesChart;
    }

    function buildCumulativeChart(cumData) {
        const ctx = document.getElementById('lii-timeseries-chart');
        if (!ctx) return;
        if (timeseriesChart) timeseriesChart.destroy();

        const labels = cumData.map(d => d.date);

        timeseriesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'LII Price Level',
                        data: cumData.map(d => d.lii_level),
                        borderColor: COLORS.lii,
                        borderWidth: 2.5,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: false,
                    },
                    {
                        label: 'CPI Price Level',
                        data: cumData.map(d => d.cpi_level),
                        borderColor: COLORS.cpi,
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: COLORS.text, usePointStyle: true } },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#374151',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'quarter', tooltipFormat: 'MMM yyyy' },
                        grid: { color: COLORS.gridLine },
                        ticks: { color: COLORS.text },
                    },
                    y: {
                        grid: { color: COLORS.gridLine },
                        ticks: { color: COLORS.text },
                        title: { display: true, text: 'Index (Jan 2020 = 100)', color: COLORS.text },
                    },
                },
            },
        });
    }

    async function addSentimentOverlay() {
        if (!timeseriesChart || showCumulative) return;
        try {
            const sData = await fetchSentiment();
            if (!sData || !sData.length) return;

            // Normalize sentiment to fit on the same Y axis range
            const yVals = timeseriesChart.data.datasets[0].data.filter(v => v !== null);
            const yMin = Math.min(...yVals);
            const yMax = Math.max(...yVals);

            const sVals = sData.map(d => d.value);
            const sMin = Math.min(...sVals);
            const sMax = Math.max(...sVals);

            const normalized = sData.map(d => ({
                x: d.date,
                y: yMin + ((d.value - sMin) / (sMax - sMin || 1)) * (yMax - yMin),
            }));

            timeseriesChart.data.datasets.push({
                label: 'Consumer Sentiment (normalized)',
                data: normalized,
                borderColor: COLORS.sentiment,
                borderWidth: 1.5,
                borderDash: [6, 3],
                pointRadius: 0,
                tension: 0.3,
                fill: false,
                yAxisID: 'y',
            });
            timeseriesChart.update();
        } catch (e) {
            console.warn('Sentiment overlay error:', e);
        }
    }

    function removeSentimentOverlay() {
        if (!timeseriesChart) return;
        // Remove sentiment dataset if present
        timeseriesChart.data.datasets = timeseriesChart.data.datasets.filter(
            ds => !ds.label.includes('Sentiment')
        );
        timeseriesChart.update();
    }

    // ── Spread Chart ──────────────────────────────────────────────────
    function buildSpreadChart(tsData) {
        const ctx = document.getElementById('lii-spread-chart');
        if (!ctx) return;

        if (spreadChart) spreadChart.destroy();

        const labels = tsData.map(d => d.date);
        const vals = tsData.map(d => d.spread);

        const datasets = [{
            label: 'Spread (LII - CPI)',
            data: vals,
            borderColor: COLORS.spread,
            backgroundColor: function (context) {
                const chart = context.chart;
                const { ctx: c, chartArea } = chart;
                if (!chartArea) return COLORS.spread + '40';
                const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, COLORS.spreadNeg + '60');
                gradient.addColorStop(0.5, COLORS.spread + '20');
                gradient.addColorStop(1, COLORS.spread + '60');
                return gradient;
            },
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: 'origin',
        }];

        // Debt spread line when overlay active
        if (debtOverlay && tsData.length > 0 && tsData[0].spread_debt != null) {
            datasets.push({
                label: 'Spread (LII+Debt - CPI)',
                data: tsData.map(d => d.spread_debt),
                borderColor: COLORS.debt,
                borderWidth: 2,
                borderDash: [6, 3],
                pointRadius: 0,
                tension: 0.3,
                fill: false,
            });
        }

        spreadChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: debtOverlay, labels: { color: COLORS.text, usePointStyle: true } },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#374151',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)}%`,
                        },
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'quarter', tooltipFormat: 'MMM yyyy' },
                        grid: { color: COLORS.gridLine },
                        ticks: { color: COLORS.text, maxTicksLimit: 12 },
                    },
                    y: {
                        grid: { color: COLORS.gridLine },
                        ticks: {
                            color: COLORS.text,
                            callback: v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%',
                        },
                    },
                },
            },
        });
    }

    // ── Category Table ────────────────────────────────────────────────
    let sortCol = 'lii_contribution';
    let sortAsc = false;

    function renderCategoryTable(categories) {
        const tbody = document.getElementById('lii-category-tbody');
        if (!tbody) return;

        // Sort
        const sorted = [...categories].sort((a, b) => {
            let va = a[sortCol], vb = b[sortCol];
            if (typeof va === 'string') va = va.toLowerCase();
            if (typeof vb === 'string') vb = vb.toLowerCase();
            if (va < vb) return sortAsc ? -1 : 1;
            if (va > vb) return sortAsc ? 1 : -1;
            return 0;
        });

        tbody.innerHTML = sorted.map(c => {
            const deltaClass = c.weight_delta > 0 ? 'lii-delta-up' : c.weight_delta < 0 ? 'lii-delta-down' : '';
            const yoyClass = c.yoy_pct >= 0 ? 'lii-val-pos' : 'lii-val-neg';
            const isDebt = c.is_debt === true;
            const rowClass = isDebt ? ' class="lii-debt-row"' : '';
            const badge = isDebt ? ' <span class="lii-noncpi-badge">NON-CPI</span>' : '';
            return `<tr${rowClass}>
                <td>${c.name}${badge}</td>
                <td class="lii-freq">${c.freq_label}</td>
                <td class="num">${c.cpi_weight.toFixed(1)}%</td>
                <td class="num">${c.lii_weight.toFixed(1)}%</td>
                <td class="num ${deltaClass}">${c.weight_delta >= 0 ? '+' : ''}${c.weight_delta.toFixed(1)}%</td>
                <td class="num ${yoyClass}">${c.yoy_pct >= 0 ? '+' : ''}${c.yoy_pct.toFixed(2)}%</td>
                <td class="num">${c.cpi_contribution.toFixed(3)}%</td>
                <td class="num">${c.lii_contribution.toFixed(3)}%</td>
            </tr>`;
        }).join('');
    }

    function setupTableSorting() {
        document.querySelectorAll('.lii-table th[data-sort]').forEach(th => {
            th.addEventListener('click', function () {
                const col = this.dataset.sort;
                if (sortCol === col) {
                    sortAsc = !sortAsc;
                } else {
                    sortCol = col;
                    sortAsc = col === 'name' || col === 'freq_label';
                }
                // Update sort indicators
                document.querySelectorAll('.lii-table th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                this.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
                // Re-render with current data
                loadCategories();
            });
        });
    }

    // ── Weight Comparison Chart ───────────────────────────────────────
    function buildWeightChart(categories) {
        const ctx = document.getElementById('lii-weight-chart');
        if (!ctx) return;

        if (weightChart) weightChart.destroy();

        // Sort by LII weight for visual impact
        const sorted = [...categories].sort((a, b) => b.lii_weight - a.lii_weight);
        const labels = sorted.map(c => c.name);
        const cpiW = sorted.map(c => c.cpi_weight);
        const liiW = sorted.map(c => c.lii_weight);

        weightChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'CPI Weight',
                        data: cpiW,
                        backgroundColor: COLORS.cpi + '80',
                        borderColor: COLORS.cpi,
                        borderWidth: 1,
                    },
                    {
                        label: 'LII Weight',
                        data: liiW,
                        backgroundColor: COLORS.lii + '80',
                        borderColor: COLORS.lii,
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: COLORS.text, usePointStyle: true } },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#374151',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)}%`,
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { color: COLORS.gridLine },
                        ticks: { color: COLORS.text, callback: v => v + '%' },
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: COLORS.text, font: { size: 11 } },
                    },
                },
            },
        });

        // Callout
        const callout = document.getElementById('lii-weight-callout');
        if (callout) {
            const usedCars = categories.find(c => c.name.includes('Used vehicles'));
            const rent = categories.find(c => c.name.includes('Rent'));
            if (usedCars && rent) {
                callout.innerHTML = `Under CPI, used cars receive <strong>${usedCars.cpi_weight.toFixed(1)}%</strong> weight. Under the Lived Inflation Index, they receive <strong>${usedCars.lii_weight.toFixed(1)}%</strong> weight — a <strong>${Math.abs(((usedCars.lii_weight - usedCars.cpi_weight) / usedCars.cpi_weight) * 100).toFixed(0)}%</strong> reduction. Meanwhile, rent moves from <strong>${rent.cpi_weight.toFixed(1)}%</strong> to <strong>${rent.lii_weight.toFixed(1)}%</strong>.`;
            }
        }
    }

    // ── Profile Switching ─────────────────────────────────────────────
    function setupProfiles() {
        const buttons = document.getElementById('lii-profile-buttons');
        const banner = document.getElementById('lii-profile-banner');
        const bannerText = document.getElementById('lii-profile-banner-text');
        const resetBtn = document.getElementById('lii-profile-reset');

        if (!buttons) return;

        buttons.addEventListener('click', async function (e) {
            const btn = e.target.closest('.lii-profile-btn');
            if (!btn) return;

            const profile = btn.dataset.profile;
            currentProfile = profile;

            // Update active state
            buttons.querySelectorAll('.lii-profile-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show/hide banner
            if (profile !== 'default' && banner) {
                banner.style.display = 'flex';
                if (bannerText) bannerText.textContent = `Viewing: ${btn.textContent.trim()}`;
            } else if (banner) {
                banner.style.display = 'none';
            }

            await loadAll();
        });

        if (resetBtn) {
            resetBtn.addEventListener('click', function () {
                currentProfile = 'default';
                buttons.querySelectorAll('.lii-profile-btn').forEach(b => b.classList.remove('active'));
                buttons.querySelector('[data-profile="default"]').classList.add('active');
                if (banner) banner.style.display = 'none';
                loadAll();
            });
        }
    }

    // ── Chart Toggle Handlers ─────────────────────────────────────────
    function setupToggles() {
        const cumToggle = document.getElementById('lii-cumulative-toggle');
        const sentToggle = document.getElementById('lii-sentiment-toggle');

        if (cumToggle) {
            cumToggle.addEventListener('change', async function () {
                showCumulative = this.checked;
                if (showCumulative) {
                    if (!cumulativeData) {
                        cumulativeData = (await fetchCumulative()).data || [];
                    }
                    buildCumulativeChart(cumulativeData);
                } else {
                    if (timeseriesRawData) {
                        buildTimeseriesChart(timeseriesRawData);
                        if (showSentiment) addSentimentOverlay();
                    }
                }
            });
        }

        if (sentToggle) {
            sentToggle.addEventListener('change', function () {
                showSentiment = this.checked;
                if (showSentiment) {
                    addSentimentOverlay();
                } else {
                    removeSentimentOverlay();
                }
            });
        }
    }

    // ── Debt Overlay Setup ───────────────────────────────────────────
    function setupDebtOverlay() {
        const masterToggle = document.getElementById('lii-debt-toggle');
        const subsContainer = document.getElementById('lii-debt-subs');
        const callout = document.getElementById('lii-debt-callout');
        const footnote = document.getElementById('lii-debt-footnote');

        if (!masterToggle) return;

        masterToggle.addEventListener('change', async function () {
            debtOverlay = this.checked;
            if (subsContainer) subsContainer.style.display = debtOverlay ? 'flex' : 'none';
            if (footnote) footnote.style.display = debtOverlay ? '' : 'none';
            await loadAll();
            if (debtOverlay) updateDebtCallout();
            else if (callout) callout.style.display = 'none';
        });

        // Sub-toggles
        document.querySelectorAll('.lii-debt-sub input[data-debt]').forEach(cb => {
            cb.addEventListener('change', async function () {
                debtCategories[this.dataset.debt] = this.checked;
                if (debtOverlay) {
                    await loadAll();
                    updateDebtCallout();
                }
            });
        });
    }

    async function updateDebtCallout() {
        const callout = document.getElementById('lii-debt-callout');
        if (!callout) return;
        try {
            const [currentData, debtInfo] = await Promise.all([
                fetchCurrent(),
                fetchDebtCurrent(),
            ]);
            const liiDebt = currentData.lii_debt;
            const lii = currentData.lii;
            const cpi = currentData.cpi;

            if (liiDebt == null) {
                callout.style.display = 'none';
                return;
            }

            const bpsVsLii = Math.round((liiDebt - lii) * 100);
            const bpsVsCpi = Math.round((liiDebt - cpi) * 100);
            const dirLii = bpsVsLii >= 0 ? 'above' : 'below';
            const dirCpi = bpsVsCpi >= 0 ? 'above' : 'below';

            const studentT = debtInfo.student_loans_T ? `$${debtInfo.student_loans_T.toFixed(2)}T` : '$1.78T';
            const creditT = debtInfo.revolving_credit_T ? `$${debtInfo.revolving_credit_T.toFixed(2)}T` : '$1.33T';
            const autoT = debtInfo.auto_loans_T ? `$${debtInfo.auto_loans_T.toFixed(2)}T` : '$1.63T';

            callout.innerHTML = `With debt servicing costs included, the Lived Inflation Index reads <strong>${liiDebt.toFixed(2)}%</strong> — <strong>${Math.abs(bpsVsLii)} bps</strong> ${dirLii} the standard LII and <strong>${Math.abs(bpsVsCpi)} bps</strong> ${dirCpi} official CPI. The BLS excludes ${studentT} in student debt, ${creditT} in revolving credit, and ${autoT} in auto loans from its inflation calculation entirely.`;
            callout.style.display = '';
        } catch (e) {
            console.warn('Debt callout error:', e);
            callout.style.display = 'none';
        }
    }

    // ── Data Loaders ──────────────────────────────────────────────────
    async function loadCategories() {
        try {
            const catData = await fetchCategories();
            const cats = catData.categories || [];
            renderCategoryTable(cats);
            buildWeightChart(cats);
        } catch (e) {
            console.error('Category load error:', e);
        }
    }

    async function loadAll() {
        try {
            // Fetch all in parallel
            const [current, tsResp, catData] = await Promise.all([
                fetchCurrent(),
                fetchTimeseries(),
                fetchCategories(),
            ]);

            // Hero
            renderHero(current);

            // Timeseries
            const tsData = tsResp.data || [];
            timeseriesRawData = tsData;

            if (showCumulative) {
                cumulativeData = (await fetchCumulative()).data || [];
                buildCumulativeChart(cumulativeData);
            } else {
                buildTimeseriesChart(tsData);
                if (showSentiment) addSentimentOverlay();
            }

            // Spread
            buildSpreadChart(tsData);

            // Categories + weight chart
            const cats = catData.categories || [];
            renderCategoryTable(cats);
            buildWeightChart(cats);
        } catch (e) {
            console.error('LII load error:', e);
            const dt = document.getElementById('lii-data-through');
            if (dt) dt.textContent = 'Error loading data';
        }
    }

    // ── Init ──────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', function () {
        setupTableSorting();
        setupProfiles();
        setupToggles();
        setupDebtOverlay();
        loadAll();
    });
})();
