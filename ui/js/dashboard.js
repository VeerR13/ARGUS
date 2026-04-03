// ═══════════════════════════════════════════════════════════
// ARGUS AI — dashboard.js
// Charts, stats, incident log, persona switcher, AI panel
// ═══════════════════════════════════════════════════════════

import {
  getAnalysis, exportCSV, exportPDF,
  streamClaudeAnalysis, askClaude,
} from './api.js';

import {
  formatTime, formatDuration, formatTimestamp,
  generateSessionId, severityColor, causalLabel,
  resizeCanvas, debounce, parseMarkdown,
} from './utils.js';

// ── State ─────────────────────────────────────────────────────

let analysisData   = null;
let activePersona  = 'insurance';
const SESSION_ID   = generateSessionId();
let allChartDrawFns = [];  // collected so we can re-draw on resize

// ── Boot ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  // Restore state from localStorage
  const videoId  = localStorage.getItem('tl_video_id')  || 'demo_001';
  const filename = localStorage.getItem('tl_filename')  || 'intersection_footage.mp4';
  const persona  = localStorage.getItem('tl_persona')   || 'insurance';
  activePersona  = persona;

  // Try to use cached analysis first, fall back to fetch
  const cached = localStorage.getItem('tl_analysis');
  try {
    analysisData = cached ? JSON.parse(cached) : await getAnalysis(videoId);
  } catch (e) {
    analysisData = await getAnalysis(videoId);
  }

  populateNav(filename);
  populateHeader(analysisData, filename);
  populateStats(analysisData);
  populateFileInfoBar(analysisData, filename);
  buildAnomalyLog(analysisData.incidents);
  buildConfidenceBars(analysisData);
  populateAIMetrics(analysisData);
  buildRecommendations(analysisData);
  updatePersonaUI(activePersona);
  initPersonaTabs();
  initCharts(analysisData);
  initIntersectionObserver();
  initGenerateButton(analysisData);
  initExportButtons(videoId, analysisData);
  initChat(analysisData);
  animateConfidenceBars(analysisData);
});

// ── Nav ───────────────────────────────────────────────────────

function populateNav(filename) {
  const el = document.getElementById('nav-filename');
  if (el) el.textContent = filename;
}

// ── Header ────────────────────────────────────────────────────

function populateHeader(data, filename) {
  const { metadata, summary } = data;

  const sessionLabel = document.getElementById('session-label');
  if (sessionLabel)
    sessionLabel.textContent = `RESEARCH OUTPUT / SESSION ${SESSION_ID}`;

  const chipDuration = document.getElementById('chip-duration');
  if (chipDuration)
    chipDuration.textContent = formatDuration(metadata.duration_seconds);

  const chipRes = document.getElementById('chip-resolution');
  if (chipRes)
    chipRes.textContent = metadata.resolution;

  const footerSession = document.getElementById('footer-session');
  if (footerSession)
    footerSession.textContent =
      `Session ${SESSION_ID} · Processed in ${summary.processing_time_seconds}s · ARGUS-v2.1`;
}

// ── File info bar ─────────────────────────────────────────────

function populateFileInfoBar(data, filename) {
  const { metadata, summary } = data;
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  safeSet('file-info-name',   filename);
  safeSet('file-fps',         `${metadata.fps} fps`);
  safeSet('file-resolution',  metadata.resolution);
  safeSet('file-processing',  `${summary.processing_time_seconds}s processing`);
}

// ── Stats ─────────────────────────────────────────────────────

function populateStats(data) {
  const { summary } = data;
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  safeSet('stat-vehicles',    summary.total_vehicles);
  safeSet('stat-speed',       summary.avg_speed_kmh);
  safeSet('stat-congestion',  summary.congestion_index.toFixed(1));
  safeSet('stat-anomalies',   summary.total_incidents);

  const critCount = data.incidents.filter(i => i.severity === 'high').length;
  safeSet('stat-critical-count', `↑ ${critCount} events critical`);
  safeSet('finding-critical', `${critCount} events`);

  // AI panel max speed & fps
  safeSet('metric-max-speed', `${summary.max_speed_kmh} km/h`);
  safeSet('metric-fps',       `${summary.fps || data.metadata.fps} fps`);

  // ── Finding 01: peak flow + time window ──────────────────────
  const timeline = summary.flow_timeline;
  const dur      = data.metadata.duration_seconds;
  const peakVal  = Math.max(...timeline);
  const peakIdx  = timeline.indexOf(peakVal);
  const secPerBin = dur / timeline.length;
  const peakStart = formatTime(peakIdx * secPerBin);
  const peakEnd   = formatTime((peakIdx + 1) * secPerBin);
  // Rolling average over all bins except the peak bin
  const otherBins = timeline.filter((_, i) => i !== peakIdx);
  const rollingAvg = otherBins.reduce((a, b) => a + b, 0) / otherBins.length;
  const peakAbove  = Math.round(((peakVal - rollingAvg) / rollingAvg) * 100);
  safeSet('finding-peak-flow',  `${peakVal} veh/min`);
  safeSet('finding-peak-label', `Peak flow at ${peakStart}–${peakEnd} window — ${peakAbove}% above rolling average`);

  // ── Finding 02: lane 2 excess ─────────────────────────────────
  const lu        = summary.lane_utilization;
  const lane2     = lu['lane2'] || lu['lane_2'] || 0;
  const others    = Object.entries(lu).filter(([k]) => k !== 'lane2' && k !== 'lane_2').map(([, v]) => v);
  const otherAvg  = others.length ? others.reduce((a, b) => a + b, 0) / others.length : 1;
  const excess    = Math.round(((lane2 - otherAvg) / otherAvg) * 100);
  const bottleneck = Object.entries(lu).reduce((a, b) => b[1] > a[1] ? b : a)[0].replace('lane', 'Lane ');
  safeSet('finding-lane-excess', `${excess}%`);
  safeSet('finding-lane-label',  `${bottleneck} occupancy excess vs other lanes — primary bottleneck identified`);
}

// ── Anomaly log ───────────────────────────────────────────────

function buildAnomalyLog(incidents) {
  const log = document.getElementById('anomaly-log');
  if (!log) return;

  log.innerHTML = incidents.map(inc => {
    const color = severityColor(inc.severity);
    const badge = `<span class="badge badge-${inc.severity}">${inc.severity.toUpperCase()}</span>`;
    return `
      <div class="anomaly-row" data-id="${inc.id}" role="button" tabindex="0"
           aria-label="View incident ${inc.id}">
        <span class="anomaly-ts">${formatTimestamp(inc.timestamp_start)}</span>
        ${badge}
        <span class="anomaly-desc">${inc.description}</span>
      </div>
    `;
  }).join('');

  log.querySelectorAll('.anomaly-row').forEach(row => {
    const open = () => window.location.href = `incident.html?id=${row.dataset.id}`;
    row.addEventListener('click', open);
    row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') open(); });
  });
}

// ── Confidence bars ───────────────────────────────────────────

function buildConfidenceBars(data) {
  const { summary } = data;
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  safeSet('conf-detection', summary.detection_confidence.toFixed(1));
  safeSet('conf-tracking',  summary.tracking_accuracy.toFixed(1));
  safeSet('conf-class',     summary.classification_precision.toFixed(1));
}

function animateConfidenceBars(data) {
  const { summary } = data;
  const bars = [
    ['conf-bar-detection', summary.detection_confidence],
    ['conf-bar-tracking',  summary.tracking_accuracy],
    ['conf-bar-class',     summary.classification_precision],
  ];
  // Slight delay so bars animate after page is visible
  setTimeout(() => {
    bars.forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el) el.style.width = `${val}%`;
    });
  }, 400);
}

// ── AI panel metrics ──────────────────────────────────────────

function populateAIMetrics(data) {
  const { summary, metadata } = data;
  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  const peak = Math.max(...summary.flow_timeline);
  safeSet('metric-peak-flow', `${peak} veh/min`);

  // Min headway: minimum time gap between vehicles at peak flow (60s / peak veh/min)
  const minHeadway = peak > 0 ? (60 / peak).toFixed(1) : '–';
  safeSet('metric-min-headway', `${minHeadway} s`);

  // Track loss rate: inverse of tracking accuracy
  const trackLoss = (100 - summary.tracking_accuracy).toFixed(1);
  safeSet('metric-track-loss', `${trackLoss}%`);

  // Avg dwell: rough estimate — video duration / total vehicles (seconds per vehicle)
  const avgDwell = summary.total_vehicles > 0
    ? (metadata.duration_seconds / summary.total_vehicles).toFixed(1)
    : '–';
  safeSet('metric-avg-dwell', `${avgDwell} s`);
}

// ── Persona tabs ──────────────────────────────────────────────

const PERSONA_LABELS = {
  insurance:  'Insurance Investigator',
  engineer:   'Traffic Engineer',
  researcher: 'Road Safety Researcher',
};

function updatePersonaUI(persona) {
  // Update tab active state
  document.querySelectorAll('.persona-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.persona === persona);
  });

  // Update AI panel badge
  const badge = document.getElementById('ai-persona-badge');
  if (badge) badge.textContent = PERSONA_LABELS[persona] || persona;
}

function initPersonaTabs() {
  document.querySelectorAll('.persona-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      activePersona = tab.dataset.persona;
      localStorage.setItem('tl_persona', activePersona);
      updatePersonaUI(activePersona);
      // Reset AI text area to placeholder
      const area = document.getElementById('ai-text-area');
      if (area) {
        area.classList.remove('streaming');
        area.innerHTML = "Click 'Generate Analysis' to produce AI-powered insights for the selected persona.";
      }
    });
  });
}

// ── Generate analysis button ──────────────────────────────────

const MOCK_AI_TEXTS = {
  insurance: `Analysis of intersection_footage.mp4 reveals two confirmed accident events and seven near-miss incidents across a 3-minute 42-second observation window. For insurance liability assessment, the primary incident at T+02:55 presents the strongest evidentiary basis — Vehicle V31 was stationary in an active lane, constituting a clear road obstruction. The contra-flow violation at T+03:18 by Vehicle V41 indicates a high-probability fault scenario with 89% model confidence. Recommend prioritising these two events for claim investigation.`,
  engineer:  `Systemic analysis of this intersection reveals persistent lane 2 saturation at 91% utilisation — 33 percentage points above lane 4. The peak flow event at T+01:40 produced a 34% density spike that cascaded into the near-miss cluster observed between T+01:22 and T+01:25. Signal timing adjustment during 30-50 veh/min inflow windows would directly reduce the TTC violations detected. Recommend geometric assessment of lane 2 approach geometry for channelisation opportunities.`,
  researcher: `The near-miss to accident ratio of 3.5:1 in this dataset is consistent with the Heinrich pyramid model at urban signalised intersections. The causal factor distribution — 38% speed differential, 29% obstruction, 22% tailgating, 11% lane change — aligns with prior literature on mixed-use arterial conflicts. Detection confidence of 94.2% and tracking accuracy of 89.7% provide sufficient data quality for statistical inference. Notable limitation: the 3-frame temporal confirmation buffer may introduce a 0.12s recall delay affecting short-duration near-miss classification.`,
};

function typewriterEffect(element, text, speed = 18) {
  return new Promise(resolve => {
    element.textContent = '';
    element.classList.add('streaming');
    let i = 0;
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    element.appendChild(cursor);

    const tick = () => {
      if (i < text.length) {
        cursor.insertAdjacentText('beforebegin', text[i]);
        i++;
        setTimeout(tick, speed);
      } else {
        cursor.remove();
        resolve();
      }
    };
    tick();
  });
}

function initGenerateButton(data) {
  const btn  = document.getElementById('btn-generate');
  const area = document.getElementById('ai-text-area');
  if (!btn || !area) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Analysing…';

    area.classList.remove('streaming');
    area.innerHTML = '';

    // Try real Claude first; on failure, fall back to mock typewriter
    let usedMock = false;
    const chunks = [];
    let streamDone = false;

    try {
      await new Promise((resolve, reject) => {
        streamClaudeAnalysis(data, activePersona,
          (chunk) => {
            if (chunk.includes('backend not connected') || chunk.includes('unavailable')) {
              usedMock = true;
            }
            chunks.push(chunk);
          },
          () => { streamDone = true; resolve(); }
        );
        // Timeout: if no response in 4s, use mock
        setTimeout(() => { if (!streamDone) { usedMock = true; resolve(); } }, 4000);
      });
    } catch (e) {
      usedMock = true;
    }

    const fullText = usedMock || chunks.join('').includes('not connected')
      ? MOCK_AI_TEXTS[activePersona] || MOCK_AI_TEXTS.insurance
      : chunks.join('');

    await typewriterEffect(area, fullText, 16);

    btn.disabled = false;
    btn.textContent = 'Regenerate Analysis';
  });
}

// ── Recommendations ───────────────────────────────────────────

function buildRecommendations(data) {
  const el = document.getElementById('ai-recommendations');
  if (!el) return;

  const { summary, incidents } = data;

  // Count causal factor types
  const causalCounts = {};
  incidents.forEach(inc => {
    inc.causal_factors.forEach(cf => {
      causalCounts[cf.type] = (causalCounts[cf.type] || 0) + 1;
    });
  });

  // Build recommendations from data
  const recs = [];

  // Most frequent causal factor
  const topCausal = Object.entries(causalCounts).sort((a, b) => b[1] - a[1])[0];
  if (topCausal) {
    const label = {
      tailgating:         'tailgating behaviour',
      speed_differential: 'speed differential violations',
      unsafe_lane_change: 'unsafe lane change incidents',
      failure_to_yield:   'failure-to-yield events',
      obstruction:        'road obstruction events',
    }[topCausal[0]] || topCausal[0];
    recs.push(`High ${label} frequency (${topCausal[1]} events) — recommend enforcement and driver awareness campaign`);
  }

  // Lane congestion
  const lu = summary.lane_utilization;
  const bottleneck = Object.entries(lu).reduce((a, b) => b[1] > a[1] ? b : a);
  if (bottleneck[1] > 80) {
    recs.push(`${bottleneck[0].replace('lane', 'Lane ')} at ${bottleneck[1]}% utilisation — install adaptive signal timing to reduce congestion`);
  }

  // High-confidence vehicles to flag
  const flagVehicles = [];
  incidents
    .filter(i => i.confidence >= 0.85 && i.vehicles_involved.length > 0)
    .slice(0, 2)
    .forEach(i => i.vehicles_involved.slice(0, 2).forEach(v => flagVehicles.push(`V${String(v).padStart(2, '0')}`)));
  const uniqueVehicles = [...new Set(flagVehicles)].slice(0, 3);
  if (uniqueVehicles.length > 0) {
    recs.push(`Flag vehicles ${uniqueVehicles.join(', ')} for follow-up — involved in high-confidence incidents`);
  }

  el.innerHTML = recs.map(r => `<div class="ai-rec-item">${r}</div>`).join('');
}

// ── Export buttons ────────────────────────────────────────────

function initExportButtons(videoId, data) {
  document.getElementById('btn-export-csv')
    ?.addEventListener('click', () => exportCSV(videoId, data));
  document.getElementById('btn-export-pdf')
    ?.addEventListener('click', () => exportPDF(videoId));
}

// ── IntersectionObserver fade-ins ─────────────────────────────

function initIntersectionObserver() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.fade-in-section').forEach(el => observer.observe(el));
}

// ═══════════════════════════════════════════════════════════
// CHARTS — raw Canvas 2D, no external libraries
// ═══════════════════════════════════════════════════════════

function initCharts(data) {
  const { summary } = data;

  const dur = data.metadata.duration_seconds;

  // Collect [canvas, drawFn, args] for resize re-draw
  const charts = [
    ['flow-chart',    drawFlowChart,    summary.flow_timeline,       data.incidents, dur],
    ['speed-chart',   drawSpeedChart,   summary.speed_distribution],
    ['donut-chart',   drawDonutChart,   summary.vehicle_composition, summary.total_vehicles],
    ['lane-chart',    drawLaneChart,    summary.lane_utilization],
    ['heatmap-chart', drawHeatmap,      summary.congestion_heatmap,  dur],
  ];

  charts.forEach(([id, fn, ...args]) => {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    fn(canvas, ...args);
    allChartDrawFns.push(() => fn(canvas, ...args));
  });

  // Resize listener
  window.addEventListener('resize', debounce(() => {
    allChartDrawFns.forEach(fn => fn());
  }, 150));
}

// ── Flow chart ────────────────────────────────────────────────

function drawFlowChart(canvas, timeline, incidents, durationSeconds) {
  const H   = 150;
  const ctx = resizeCanvas(canvas, H);
  const W   = canvas.offsetWidth;
  const PAD = { top: 16, right: 16, bottom: 28, left: 36 };
  const cW  = W - PAD.left - PAD.right;
  const cH  = H - PAD.top  - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...timeline) * 1.15;
  const n      = timeline.length;

  const xOf = (i) => PAD.left + (i / (n - 1)) * cW;
  const yOf = (v) => PAD.top  + cH - (v / maxVal) * cH;

  // Grid lines
  const gridLines = 4;
  ctx.strokeStyle = 'rgba(74,158,255,0.07)';
  ctx.lineWidth   = 1;
  for (let g = 0; g <= gridLines; g++) {
    const y = PAD.top + (g / gridLines) * cH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
    const label = Math.round(maxVal - (g / gridLines) * maxVal);
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'right';
    ctx.fillText(label, PAD.left - 6, y + 3);
  }

  // Derive anomaly indices from incident timestamps
  const dur = durationSeconds || 222;
  const anomalyIndices = (incidents || [])
    .map(inc => Math.round((inc.timestamp_start / dur) * (n - 1)))
    .filter(ai => ai >= 0 && ai < n);

  anomalyIndices.forEach(ai => {
    const x = xOf(ai);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,77,109,0.55)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + cH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  });

  // Area gradient fill
  const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + cH);
  grad.addColorStop(0,   'rgba(74,158,255,0.22)');
  grad.addColorStop(1,   'rgba(74,158,255,0)');

  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(timeline[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(timeline[i]));
  ctx.lineTo(xOf(n - 1), PAD.top + cH);
  ctx.lineTo(xOf(0),     PAD.top + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#4a9eff';
  ctx.lineWidth   = 1.5;
  ctx.moveTo(xOf(0), yOf(timeline[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(timeline[i]));
  ctx.stroke();

  // Anomaly red dots
  anomalyIndices.forEach(ai => {
    const x = xOf(ai);
    const y = yOf(timeline[ai]);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff4d6d';
    ctx.fill();
  });

  // X-axis time labels
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font      = '7px "Share Tech Mono"';
  ctx.textAlign = 'center';
  const step = Math.floor(n / 6);
  for (let i = 0; i < n; i += step) {
    const totalSecs = (i / (n - 1)) * dur;
    ctx.fillText(formatTime(totalSecs), xOf(i), H - 8);
  }
}

// ── Speed distribution chart ──────────────────────────────────

function drawSpeedChart(canvas, distribution) {
  const H    = 190;
  const ctx  = resizeCanvas(canvas, H);
  const W    = canvas.offsetWidth;
  const PAD  = { top: 10, right: 60, bottom: 10, left: 56 };
  const cW   = W - PAD.left - PAD.right;
  const cH   = H - PAD.top  - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  const labels = ['0–20', '20–40', '40–60', '60–80', '80+'];
  const keys   = ['0-20', '20-40', '40-60', '60-80', '80+'];
  const vals   = keys.map(k => distribution[k] || 0);
  const maxVal = Math.max(...vals);
  const peak   = vals.indexOf(maxVal);

  const barH   = (cH / labels.length) * 0.55;
  const gapH   = (cH / labels.length);

  labels.forEach((lbl, i) => {
    const y      = PAD.top + i * gapH + (gapH - barH) / 2;
    const barW   = (vals[i] / maxVal) * cW;
    const isPeak = i === peak;

    // Bar gradient
    const grad = ctx.createLinearGradient(PAD.left, 0, PAD.left + cW, 0);
    if (isPeak) {
      grad.addColorStop(0, '#00e5ff');
      grad.addColorStop(1, 'rgba(0,229,255,0.15)');
    } else {
      grad.addColorStop(0, '#4a9eff');
      grad.addColorStop(1, 'rgba(74,158,255,0.08)');
    }

    ctx.fillStyle = grad;
    ctx.fillRect(PAD.left, y, barW, barH);

    // Label (left)
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'right';
    ctx.fillText(lbl, PAD.left - 8, y + barH / 2 + 3);

    // Value (right of bar)
    ctx.fillStyle = isPeak ? '#00e5ff' : 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText(vals[i], PAD.left + barW + 6, y + barH / 2 + 3);
  });
}

// ── Donut chart ───────────────────────────────────────────────

function drawDonutChart(canvas, composition, total) {
  const H   = 190;
  const ctx = resizeCanvas(canvas, H);
  const W   = canvas.offsetWidth;
  ctx.clearRect(0, 0, W, H);

  const COLORS = ['#4a9eff', '#00e5ff', '#ff9f43', '#ff4d6d', '#a78bfa'];
  const labels = Object.keys(composition);
  const vals   = Object.values(composition);
  const sum    = vals.reduce((a, b) => a + b, 0);

  // Chart on the left half
  const cx = W * 0.38;
  const cy = H / 2;
  const r  = Math.min(cx, cy) * 0.72;
  const ri = r * 0.58;  // inner radius (donut hole)

  let angle = -Math.PI / 2;
  vals.forEach((val, i) => {
    const slice = (val / sum) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fill();
    angle += slice;
  });

  // Dark hole
  ctx.beginPath();
  ctx.arc(cx, cy, ri, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(10,21,32,0.95)';
  ctx.fill();

  // Center text
  ctx.fillStyle = '#fff';
  ctx.font      = '300 26px "Cormorant Garamond"';
  ctx.textAlign = 'center';
  ctx.fillText(total, cx, cy + 4);
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font      = '7px "Share Tech Mono"';
  ctx.fillText('total', cx, cy + 16);

  // Legend
  const legendX = W * 0.65;
  const itemH   = H / (labels.length + 1);
  labels.forEach((lbl, i) => {
    const y = (i + 0.5) * itemH + itemH * 0.5;
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fillRect(legendX, y - 5, 8, 8);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'left';
    ctx.fillText(lbl.toUpperCase(), legendX + 14, y + 3);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.textAlign = 'right';
    ctx.fillText(vals[i], W - 8, y + 3);
  });
}

// ── Lane utilization chart ────────────────────────────────────

function drawLaneChart(canvas, utilization) {
  const H   = 190;
  const ctx = resizeCanvas(canvas, H);
  const W   = canvas.offsetWidth;
  const PAD = { top: 30, right: 14, bottom: 28, left: 14 };
  const cW  = W - PAD.left - PAD.right;
  const cH  = H - PAD.top  - PAD.bottom;
  ctx.clearRect(0, 0, W, H);

  const labels = Object.keys(utilization);
  const vals   = Object.values(utilization);
  const n      = labels.length;
  const barW   = (cW / n) * 0.55;
  const gap    = cW / n;

  vals.forEach((val, i) => {
    const x    = PAD.left + i * gap + (gap - barW) / 2;
    const barH = (val / 100) * cH;
    const y    = PAD.top + cH - barH;

    // Color by utilization level
    const color = val > 85 ? '#ff4d6d' : val > 70 ? '#ff9f43' : '#7cffcb';
    const grad  = ctx.createLinearGradient(0, y, 0, y + barH);
    grad.addColorStop(0, color);
    grad.addColorStop(1, `${color}22`);

    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW, barH);

    // Percentage label above bar
    ctx.fillStyle = color;
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'center';
    ctx.fillText(`${val}%`, x + barW / 2, y - 6);

    // Lane label below
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font      = '7px "Share Tech Mono"';
    ctx.fillText(labels[i].replace('lane', 'L'), x + barW / 2, H - 8);
  });

  // Grid lines
  ctx.strokeStyle = 'rgba(74,158,255,0.07)';
  ctx.lineWidth   = 1;
  [25, 50, 75, 100].forEach(pct => {
    const y = PAD.top + cH - (pct / 100) * cH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
  });
}

// ── Congestion heatmap ────────────────────────────────────────

function drawHeatmap(canvas, heatmap, durationSeconds) {
  const H      = 160;
  const ctx    = resizeCanvas(canvas, H);
  const W      = canvas.offsetWidth;
  const PAD    = { top: 12, right: 40, bottom: 24, left: 28 };
  const cW     = W - PAD.left - PAD.right;
  const cH     = H - PAD.top  - PAD.bottom;
  const rows   = heatmap.length;
  const cols   = heatmap[0].length;
  const cellW  = cW / cols;
  const cellH  = cH / rows;
  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...heatmap.flat());

  function heatColor(v, max) {
    const t = v / max;
    if (t < 0.5) {
      const s = t / 0.5;
      return `rgba(${Math.round(30 + s * 150)}, ${Math.round(80 + s * 40)}, ${Math.round(180 - s * 60)}, ${0.5 + s * 0.4})`;
    } else {
      const s = (t - 0.5) / 0.5;
      return `rgba(${Math.round(180 + s * 40)}, ${Math.round(120 - s * 80)}, ${Math.round(120 - s * 80)}, ${0.7 + s * 0.3})`;
    }
  }

  heatmap.forEach((row, ri) => {
    row.forEach((val, ci) => {
      const x = PAD.left + ci * cellW;
      const y = PAD.top  + ri * cellH;
      ctx.fillStyle = heatColor(val, maxVal);
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
    });
  });

  // Row labels
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font      = '7px "Share Tech Mono"';
  ctx.textAlign = 'right';
  for (let ri = 0; ri < rows; ri++) {
    const y = PAD.top + ri * cellH + cellH / 2 + 3;
    ctx.fillText(`L${ri + 1}`, PAD.left - 4, y);
  }

  // Col labels (time)
  ctx.textAlign = 'center';
  const step = Math.floor(cols / 5);
  const dur  = durationSeconds || 222;
  for (let ci = 0; ci < cols; ci += step) {
    const x          = PAD.left + ci * cellW + cellW / 2;
    const totalSecs  = (ci / cols) * dur;
    ctx.fillText(formatTime(totalSecs), x, H - 6);
  }

  // Colour legend (right side)
  const lgX = W - PAD.right + 6;
  const lgH = cH;
  const lgW = 8;
  const lgGrad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + lgH);
  lgGrad.addColorStop(0,   'rgba(220,40,40,0.9)');
  lgGrad.addColorStop(0.5, 'rgba(180,120,20,0.8)');
  lgGrad.addColorStop(1,   'rgba(30,80,180,0.7)');
  ctx.fillStyle = lgGrad;
  ctx.fillRect(lgX, PAD.top, lgW, lgH);

  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font      = '6px "Share Tech Mono"';
  ctx.textAlign = 'left';
  ctx.fillText('HI', lgX + lgW + 3, PAD.top + 6);
  ctx.fillText('LO', lgX + lgW + 3, PAD.top + lgH - 2);
}

// ═══════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════

function initChat(data) {
  const toggleBtn  = document.getElementById('chat-toggle');
  const closeBtn   = document.getElementById('chat-close');
  const panel      = document.getElementById('chat-panel');
  const input      = document.getElementById('chat-input');
  const sendBtn    = document.getElementById('chat-send');
  const messages   = document.getElementById('chat-messages');
  const suggestions = document.getElementById('chat-suggestions');

  if (!toggleBtn || !panel) return;

  toggleBtn.addEventListener('click', () => {
    const open = panel.classList.toggle('chat-open');
    panel.style.display = open ? 'flex' : 'none';
    if (open) input?.focus();
  });

  closeBtn?.addEventListener('click', () => {
    panel.style.display = 'none';
  });

  function addMessage(text, isUser) {
    const div = document.createElement('div');
    div.style.cssText = isUser
      ? 'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);padding:10px 12px;font-family:"Barlow",sans-serif;font-weight:300;font-size:12px;color:rgba(255,255,255,0.7);text-align:right;line-height:1.6;'
      : 'background:rgba(74,158,255,0.06);border-left:2px solid #4a9eff;padding:10px 12px;font-family:"Barlow",sans-serif;font-weight:300;font-size:12px;color:rgba(255,255,255,0.6);line-height:1.7;';
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  async function sendMessage(question) {
    if (!question.trim()) return;

    // Hide suggestions on first question
    if (suggestions) suggestions.style.display = 'none';

    addMessage(question, true);
    const aiDiv = addMessage('', false);
    aiDiv.textContent = '';

    // Streaming response
    await new Promise(resolve => {
      askClaude(question, data,
        (chunk) => {
          aiDiv.textContent += chunk;
          messages.scrollTop = messages.scrollHeight;
        },
        resolve
      );
    });
  }

  sendBtn?.addEventListener('click', () => {
    sendMessage(input.value);
    input.value = '';
  });

  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      sendMessage(input.value);
      input.value = '';
    }
  });

  // Suggestion pills
  document.querySelectorAll('.chat-suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
      sendMessage(btn.dataset.q);
    });
  });
}
