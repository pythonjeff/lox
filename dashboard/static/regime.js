/* ───────────────────────────────────────────────────────────────
   regime.js  — LOX FUND regime page renderer (v3 — actionable)
   ─────────────────────────────────────────────────────────────── */

// Helpers ──────────────────────────────────────────────────────

// Severity → gauge/badge color. Muted palette, not neon.
const SEVERITY_COLORS = {
  low:      { text: "#64d28a", bg: "rgba(100,210,138,0.10)" },
  moderate: { text: "#94a3b8", bg: "rgba(148,163,184,0.10)" },
  elevated: { text: "#e0a845", bg: "rgba(224,168,69,0.10)" },
  high:     { text: "#e05555", bg: "rgba(224,85,85,0.12)" },
  neutral:  { text: "#64748b", bg: "rgba(100,116,139,0.10)" },
};

function severityColor(severity) {
  return SEVERITY_COLORS[severity] || SEVERITY_COLORS.neutral;
}

function scoreColor(score) {
  if (score == null) return { ...SEVERITY_COLORS.neutral, cls: "neutral" };
  if (score <= 30) return { ...SEVERITY_COLORS.low, cls: "low" };
  if (score <= 50) return { ...SEVERITY_COLORS.moderate, cls: "moderate" };
  if (score <= 70) return { ...SEVERITY_COLORS.elevated, cls: "elevated" };
  return { ...SEVERITY_COLORS.high, cls: "high" };
}

function regimeProximity(score, thresholds) {
  if (score == null || !thresholds || !thresholds.length) return null;
  let nearestDist = Infinity;
  let nearestLabel = "";
  for (const t of thresholds) {
    const dist = Math.abs(score - t.score);
    if (dist < nearestDist) {
      nearestDist = dist;
      // Show what you'd flip TO if you crossed this boundary
      nearestLabel = score >= t.score ? t.label_below : t.label_above;
    }
  }
  return { distance: Math.round(nearestDist), label: nearestLabel };
}

function el(id) { return document.getElementById(id); }
function show(id) { const e = el(id); if (e) e.style.display = ""; }
function hide(id) { const e = el(id); if (e) e.style.display = "none"; }


// 1. HERO ZONE ─────────────────────────────────────────────────

function renderHero(data) {
  el("rg-hero-title").textContent = data.domain || data.regime_name;
  el("rg-hero-asof").textContent = data.as_of || "—";

  const blurbEl = el("rg-hero-blurb");
  if (blurbEl) blurbEl.style.display = "none";

  const score = data.composite_score;
  const severity = data.severity || "moderate";
  const color = severityColor(severity);

  // Score block — the dominant visual element
  const scoreNum = el("rg-hero-score-num");
  if (scoreNum) scoreNum.textContent = score != null ? Math.round(score) : "—";

  const scoreBlock = el("rg-hero-score-block");
  if (scoreBlock) {
    scoreBlock.style.borderColor = color.text;
    scoreBlock.style.setProperty("--score-accent", color.text);
  }

  // Severity glow behind score
  const glow = el("rg-hero-glow");
  if (glow) glow.style.background = `radial-gradient(ellipse at 8% 50%, ${color.text}22 0%, transparent 60%)`;

  const badge = el("rg-hero-badge");
  badge.textContent = data.classification || "—";
  badge.style.background = color.bg;
  badge.style.color = color.text;
  badge.style.borderColor = color.text;

  const hero = document.querySelector(".rg-hero");
  hero.style.setProperty("--hero-accent", color.text);

  // Score delta
  const deltaEl = el("rg-score-delta");
  if (deltaEl && score != null && data.prev_score != null) {
    const delta = Math.round(score - data.prev_score);
    if (delta !== 0) {
      const arrow = delta > 0 ? "▲" : "▼";
      const deltaColor = delta > 0 ? "#ef4444" : "#22c55e";
      deltaEl.innerHTML = `<span style="color:${deltaColor}">${arrow}${Math.abs(delta)}</span>`;
      deltaEl.style.display = "";
    } else {
      deltaEl.style.display = "none";
    }
  } else if (deltaEl) {
    deltaEl.style.display = "none";
  }

  renderSpectrum(data);
}

function renderSpectrum(data) {
  const wrap = el("rg-spectrum-wrap");
  if (!wrap) return;

  const score = data.composite_score;
  const thresholds = data.thresholds || [];

  if (score == null || !thresholds.length) {
    wrap.innerHTML = "";
    return;
  }

  const edges = [0, ...thresholds.map(t => t.score), 100];
  const zones = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const lo = edges[i];
    const hi = edges[i + 1];
    let label;
    if (i === 0) {
      label = thresholds[0].label_below;
    } else {
      label = thresholds[i - 1].label_above;
    }
    const isActive = score >= lo && (i === edges.length - 2 ? score <= hi : score < hi);
    zones.push({ lo, hi, label, width: hi - lo, isActive });
  }

  const ZONE_COLORS = [
    "rgba(34,197,94,0.20)",
    "rgba(52,211,153,0.14)",
    "rgba(148,163,184,0.10)",
    "rgba(224,168,69,0.14)",
    "rgba(239,68,68,0.18)",
  ];

  let html = '<div class="rg-spectrum">';
  zones.forEach((z, i) => {
    const bg = ZONE_COLORS[Math.min(i, ZONE_COLORS.length - 1)];
    const activeClass = z.isActive ? " rg-spectrum-zone--active" : "";
    html += `<div class="rg-spectrum-zone${activeClass}" style="width:${z.width}%;background:${bg};">
      <span class="rg-spectrum-zone-label">${z.label}</span>
    </div>`;
  });
  html += '</div>';

  const markerPct = Math.max(0, Math.min(100, score));
  html += `<div class="rg-spectrum-marker" style="left:${markerPct}%;">
    <div class="rg-spectrum-marker-needle"></div>
    <div class="rg-spectrum-marker-head"></div>
  </div>`;

  for (const t of thresholds) {
    html += `<div class="rg-spectrum-tick" style="left:${t.score}%;"></div>`;
  }

  wrap.innerHTML = html;
}


// 2. METRICS TABLE ─────────────────────────────────────────────

function computeMetricStats(m) {
  const raw = m.raw_value;
  const rng = m.range;
  if (!rng || raw == null) return { pct: 50, stressPct: 50, dotColor: "#64748b" };

  const inv = rng.inverted;
  // Natural number line: lo = smaller numeric value, hi = larger
  const lo = Math.min(rng.healthy, rng.stressed);
  const hi = Math.max(rng.healthy, rng.stressed);
  const span = hi - lo;

  // pct: position on the natural scale (0% = lo end, 100% = hi end)
  let pct = span !== 0 ? ((raw - lo) / span) * 100 : 50;
  pct = Math.max(0, Math.min(100, pct));

  // stressPct: how stressed is this metric (0 = healthy, 100 = worst)
  // For non-inverted (higher=worse): stress grows with pct
  // For inverted (higher=better): stress grows as pct falls
  const stressPct = inv ? (100 - pct) : pct;

  const dotColor = stressPct > 65 ? "#e05555" : "#64748b";

  return { pct, stressPct, dotColor };
}

function signalBadge(m) {
  const sp = Math.round(m.stressPct);
  const delta = m.delta;
  const inv = m.range && m.range.inverted;

  // Severity zone label + CSS class
  let label, cls;
  if (sp >= 85)      { label = "EXTREME";  cls = "rg-sig--extreme"; }
  else if (sp >= 70) { label = "STRESS";   cls = "rg-sig--stress"; }
  else if (sp >= 55) { label = "ELEVATED"; cls = "rg-sig--elevated"; }
  else if (sp >= 40) { label = "WATCH";    cls = "rg-sig--watch"; }
  else if (sp >= 20) { label = "NORMAL";   cls = "rg-sig--normal"; }
  else               { label = "HEALTHY";  cls = "rg-sig--healthy"; }

  // Directional arrow from delta (if available)
  let arrow = "";
  if (delta != null && delta !== 0) {
    // Determine if this delta moves toward stress or health
    const towardStress = inv ? (delta < 0) : (delta > 0);
    const arrowChar = towardStress ? "▲" : "▼";
    const arrowCls = towardStress ? "rg-arrow--worse" : "rg-arrow--better";
    // Format delta magnitude compactly
    const absDelta = Math.abs(delta);
    let dStr;
    if (absDelta >= 100) dStr = absDelta.toFixed(0);
    else if (absDelta >= 1) dStr = absDelta.toFixed(1);
    else dStr = absDelta.toFixed(2);
    arrow = `<span class="rg-sig-arrow ${arrowCls}">${arrowChar}${dStr}</span>`;
  }

  return `<span class="rg-signal ${cls}">${label}</span>${arrow}`;
}

function renderMetrics(data) {
  const metrics = data.metrics || [];
  const tbody = el("rg-metrics-tbody");

  const liveMetrics = metrics.filter(m => m.value !== "—" && m.raw_value != null);

  if (!liveMetrics.length) {
    hide("rg-metrics-table-wrap");
    show("rg-metrics-empty");
    return;
  }
  show("rg-metrics-table-wrap");
  hide("rg-metrics-empty");

  // Pre-compute stats and sort by stress (worst first)
  const enriched = liveMetrics.map(m => ({ ...m, ...computeMetricStats(m) }));
  enriched.sort((a, b) => b.stressPct - a.stressPct);

  let html = "";
  for (const m of enriched) {
    const rng = m.range;
    const isStressed = m.stressPct > 65;

    const leftLabel = rng.healthy < rng.stressed ? rng.healthy_label : rng.stressed_label;
    const rightLabel = rng.healthy < rng.stressed ? rng.stressed_label : rng.healthy_label;

    const gradientClass = rng.inverted ? " rg-scale--inverted" : "";
    const rangeHtml = `<div class="rg-scale${gradientClass}">
      <div class="rg-scale-bar">
        <div class="rg-scale-dot" style="left:${m.pct}%;background:${m.dotColor};"></div>
      </div>
      <div class="rg-scale-labels">
        <span>${leftLabel}</span>
        <span>${rightLabel}</span>
      </div>
    </div>`;

    const weightPct = m.weight != null ? `${(m.weight * 100).toFixed(0)}%` : "";
    const rowClass = isStressed ? "rg-row--stressed" : "";

    html += `<tr class="${rowClass}">
      <td class="rg-td-name"><span class="rg-metric-name">${m.name}</span>${m.desc ? `<span class="rg-metric-desc">${m.desc}</span>` : ''}</td>
      <td class="rg-td-val">${m.value}</td>
      <td class="rg-td-range">${rangeHtml}</td>
      <td class="rg-td-signal">${signalBadge(m)}</td>
      <td class="rg-td-weight">${weightPct}</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}


// 3. PILLAR CHART (Chart.js) ───────────────────────────────────

let pillarChart = null;

function renderPillars(data) {
  const pillars = data.pillars || [];
  if (!pillars.length) {
    hide("rg-pillars-section");
    return;
  }
  show("rg-pillars-section");

  const labels = pillars.map(p => p.name);
  const scores = pillars.map(p => p.score || 0);
  const weights = pillars.map(p => p.weight || 0);
  const contributions = pillars.map(p => p.weighted_contribution || 0);
  const bgColors = scores.map(s => scoreColor(s).bg.replace(/[\d.]+\)$/, "0.55)"));
  const borderColors = scores.map(s => scoreColor(s).text);

  const ctx = el("rg-pillar-chart").getContext("2d");

  if (pillarChart) pillarChart.destroy();

  pillarChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Score",
        data: scores,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 4,
        barPercentage: 0.7,
        categoryPercentage: 0.8,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0, max: 100,
          grid: { color: "rgba(255,255,255,0.06)", drawTicks: false },
          ticks: {
            color: "#94a3b8",
            font: { family: "'Inter', sans-serif", size: 11 },
            callback: v => v,
          },
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#e2e8f0",
            font: { family: "'Inter', sans-serif", size: 12, weight: 500 },
            padding: 8,
          },
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          titleColor: "#f8fafc",
          bodyColor: "#cbd5e1",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 12,
          titleFont: { family: "'Inter', sans-serif", weight: 600 },
          bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
          callbacks: {
            afterLabel: function(context) {
              const i = context.dataIndex;
              const w = (weights[i] * 100).toFixed(0);
              const c = contributions[i].toFixed(1);
              return `Weight: ${w}%  |  Contribution: ${c}`;
            }
          }
        },
        annotation: {
          annotations: {
            line50: {
              type: "line",
              xMin: 50, xMax: 50,
              borderColor: "rgba(234,179,8,0.4)",
              borderWidth: 1,
              borderDash: [4, 4],
              label: { display: false },
            },
            line75: {
              type: "line",
              xMin: 75, xMax: 75,
              borderColor: "rgba(239,68,68,0.4)",
              borderWidth: 1,
              borderDash: [4, 4],
              label: { display: false },
            }
          }
        }
      },
    }
  });

  const sorted = [...pillars].sort((a, b) => (b.weighted_contribution || 0) - (a.weighted_contribution || 0));
  const top3 = sorted.slice(0, 3);
  if (top3.length) {
    show("rg-key-drivers");
    const list = el("rg-key-drivers-list");
    list.innerHTML = top3.map(p => {
      const color = scoreColor(p.score);
      return `<li>
        <span class="rg-kd-name">${p.name}</span>
        <span class="rg-kd-score" style="color:${color.text};">${p.score?.toFixed(1)}</span>
        <span class="rg-kd-contrib">→ ${p.weighted_contribution?.toFixed(1)} weighted</span>
      </li>`;
    }).join("");
  } else {
    hide("rg-key-drivers");
  }
}


// MAIN LOAD ────────────────────────────────────────────────────

function renderAll(data) {
  renderHero(data);
  renderMetrics(data);
  renderPillars(data);
}

function loadRegimeData(refresh) {
  const page = document.querySelector(".regime-page");
  const regimeName = page?.dataset?.regime;
  if (!regimeName) return;

  show("regime-loading");
  hide("regime-error");
  hide("regime-content");

  const qs = refresh ? "?refresh=true" : "";
  fetch(`/api/regime/${regimeName}${qs}`)
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      if (data.error && Object.keys(data).length === 1) {
        el("regime-error-msg").textContent = data.error;
        hide("regime-loading");
        show("regime-error");
        return;
      }
      hide("regime-loading");
      show("regime-content");
      renderAll(data);
    })
    .catch(err => {
      el("regime-error-msg").textContent = err.message;
      hide("regime-loading");
      show("regime-error");
    });
}

// Boot
document.addEventListener("DOMContentLoaded", () => {
  loadRegimeData(false);

  const refreshBtn = el("regime-refresh-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      refreshBtn.disabled = true;
      refreshBtn.classList.add("rg-hero-refresh--loading");
      loadRegimeData(true);
      setTimeout(() => {
        refreshBtn.disabled = false;
        refreshBtn.classList.remove("rg-hero-refresh--loading");
      }, 3000);
    });
  }

  // Weights column toggle
  const weightsToggle = el("rg-weights-toggle");
  if (weightsToggle) {
    weightsToggle.addEventListener("click", () => {
      const table = el("rg-metrics-table");
      if (table) {
        table.classList.toggle("rg-show-weights");
        weightsToggle.textContent = table.classList.contains("rg-show-weights") ? "Hide Weights" : "Show Weights";
      }
    });
  }
});
