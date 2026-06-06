// ═══════════════════════════════════════════════════════════
// ARGUS — dashboard.js
// Self-contained. No imports from api.js or utils.js.
// Handles: data loading, incident log, video modal,
//          flow chart, donut chart, confidence bars.
// ═══════════════════════════════════════════════════════════

// ── constants ─────────────────────────────────────────────

const API_BASE =
  localStorage.getItem('argus_api_base') ||
  'https://raghuvanshi-veer004--argus-web.modal.run';

const DONUT_COLORS = {
  car:        '#D4915A',
  motorcycle: '#A8C5A0',
  truck:      '#6B5E4E',
  bus:        '#EDE5D8',
  bicycle:    '#4a9eff',
  other:      '#3a3530',
};

const SEV_COLORS = {
  high:   '#C4615A',
  medium: '#D4915A',
  low:    '#A8C5A0',
};

const TYPE_LABELS = {
  accident:           'Collision Detected',
  near_miss:          'Near Miss',
  risky_interaction:  'Risky Interaction',
};

// ── helpers ────────────────────────────────────────────────

function fmtTimestamp(ms) {
  const secs = Math.floor(ms / 1000);
  const m    = Math.floor(secs / 60);
  const s    = String(secs % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function fmtDuration(secs) {
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

function fmtTTC(val) {
  if (val === null || val === undefined) return '—';
  return `${Number(val).toFixed(1)}s`;
}

function fmtSpeed(val) {
  if (val === null || val === undefined) return '—';
  return `${Math.round(val)} km/h`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function showError(message) {
  const page = document.getElementById('page');
  if (!page) return;
  page.innerHTML = `
    <div style="
      padding: 120px 40px;
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
      color: rgba(237,229,216,.3);
      font-size: 12px;
      letter-spacing: .12em;
      line-height: 2;
    ">
      <div style="font-size:28px;color:rgba(212,145,90,.2);margin-bottom:20px;">⚠</div>
      ${message}<br><br>
      <a href="index.html" style="
        color: #D4915A;
        border-bottom: 1px solid rgba(212,145,90,.3);
        padding-bottom: 2px;
      ">← Upload a new video</a>
    </div>
  `;
}

// ── data loading ───────────────────────────────────────────

async function loadAnalysis(videoId) {
  // 1. Try localStorage cache first
  const cached = localStorage.getItem('tl_analysis');
  if (cached) {
    try {
      const parsed = JSON.parse(cached);
      if (parsed && parsed.metadata) return parsed;
    } catch (_) {
      // corrupted cache, fall through to fetch
    }
  }

  // 2. Fall back to API fetch
  const url = `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/analysis`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`API ${resp.status}: ${resp.statusText}`);
  return resp.json();
}

// ── nav ────────────────────────────────────────────────────

function populateNav(filename) {
  const el = document.getElementById('nav-filename');
  if (el) el.textContent = filename || '—';
}

// ── stats bar ──────────────────────────────────────────────

function populateStats(data) {
  const { summary, metadata } = data;

  setText('stat-vehicles', summary.total_vehicles ?? '—');
  setText('stat-incidents', summary.total_incidents ?? '—');
  setText('stat-speed',
    summary.avg_speed_kmh != null
      ? `${Math.round(summary.avg_speed_kmh)} km/h`
      : '—'
  );
  setText('stat-duration',
    metadata.duration_seconds != null
      ? fmtDuration(metadata.duration_seconds)
      : '—'
  );
}

// ── incident log ───────────────────────────────────────────

function buildIncidentLog(incidents, videoId, onIncidentClick) {
  const container = document.getElementById('anomaly-log');
  if (!container) return;

  if (!incidents || incidents.length === 0) {
    container.innerHTML = `
      <div class="no-incidents">
        <span class="no-incidents-icon" aria-hidden="true">✓</span>
        <span class="no-incidents-text">No incidents detected</span>
      </div>
    `;
    return;
  }

  const fragment = document.createDocumentFragment();

  incidents.forEach((incident, idx) => {
    const {
      incident_id,
      type,
      severity,
      timestamp_start_ms,
      vehicles_involved = [],
      metrics = {},
    } = incident;

    const ts         = fmtTimestamp(timestamp_start_ms ?? 0);
    const typeLabel  = TYPE_LABELS[type] || type;
    const sevClass   = (severity || 'low').toLowerCase();
    const sevColor   = SEV_COLORS[sevClass] || SEV_COLORS.low;

    const vehicleText = vehicles_involved.length
      ? `Vehicles: ${vehicles_involved.map(v => `V${v}`).join(', ')}`
      : '';

    const row = document.createElement('div');
    row.className = 'incident-row';
    row.setAttribute('role', 'listitem');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', `${typeLabel}, ${severity} severity at ${ts}`);
    row.style.borderLeftColor = sevColor;

    row.innerHTML = `
      <span class="incident-ts" aria-hidden="true">${ts}</span>
      <div class="incident-body">
        <div class="incident-type-line">
          <span class="incident-type-label">${typeLabel}</span>
          <span class="sev-badge ${sevClass}">${severity}</span>
        </div>
        ${vehicleText
          ? `<div class="incident-vehicles">${vehicleText}</div>`
          : ''}
      </div>
      <span class="incident-play-hint" aria-hidden="true">&#9654; Play</span>
    `;

    const handleActivate = () => onIncidentClick(incident, videoId);
    row.addEventListener('click', handleActivate);
    row.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleActivate();
      }
    });

    fragment.appendChild(row);
  });

  container.appendChild(fragment);
}

// ── video modal ────────────────────────────────────────────

let modalVideoSrc = null;  // track currently loaded src to avoid redundant reloads

function openModal(incident, videoId) {
  const modal    = document.getElementById('video-modal');
  const video    = document.getElementById('modal-video');
  if (!modal || !video) return;

  const streamUrl = `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/stream`;
  const seekTo    = (incident.timestamp_start_ms ?? 0) / 1000;

  // only reload src if it changed
  if (video.src !== streamUrl || modalVideoSrc !== streamUrl) {
    video.src    = streamUrl;
    modalVideoSrc = streamUrl;
    video.load();
    video.addEventListener('loadedmetadata', () => {
      video.currentTime = seekTo;
    }, { once: true });
  } else {
    video.currentTime = seekTo;
  }

  // populate metrics grid
  const metrics   = incident.metrics || {};
  const typeLabel = TYPE_LABELS[incident.type] || incident.type || '—';
  const severity  = incident.severity || '—';
  const ttcText   = fmtTTC(metrics.min_ttc);
  const speedText = fmtSpeed(metrics.relative_speed_kmh);

  setText('mm-type',     typeLabel);
  setText('mm-severity', severity);
  setText('mm-ttc',      ttcText);
  setText('mm-speed',    speedText);

  // color severity value
  const mmSev = document.getElementById('mm-severity');
  if (mmSev) {
    mmSev.style.color = SEV_COLORS[(severity || '').toLowerCase()] || 'var(--text)';
  }

  modal.classList.add('open');
  document.body.style.overflow = 'hidden';

  // focus the close button for accessibility
  const closeBtn = document.getElementById('modal-close');
  if (closeBtn) closeBtn.focus();
}

function closeModal() {
  const modal = document.getElementById('video-modal');
  const video = document.getElementById('modal-video');
  if (!modal) return;

  modal.classList.remove('open');
  document.body.style.overflow = '';

  if (video) {
    video.pause();
    // don't reset src — let browser keep buffer for re-opens
  }
}

function initModalControls() {
  const modal    = document.getElementById('video-modal');
  const closeBtn = document.getElementById('modal-close');
  if (!modal) return;

  closeBtn && closeBtn.addEventListener('click', closeModal);

  // click outside modal-inner to close
  modal.addEventListener('click', e => {
    if (e.target === modal) closeModal();
  });

  // Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('open')) closeModal();
  });
}

// ── flow chart (Canvas 2D area chart) ─────────────────────

function drawFlowChart(summary, metadata) {
  const canvas = document.getElementById('flow-chart');
  if (!canvas) return;

  const timeline = summary.flow_timeline;
  if (!timeline || !timeline.length) return;

  const dpr   = window.devicePixelRatio || 1;
  const w     = canvas.offsetWidth;
  const h     = 140;

  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const PAD_L = 32, PAD_R = 12, PAD_T = 12, PAD_B = 28;
  const chartW = w - PAD_L - PAD_R;
  const chartH = h - PAD_T - PAD_B;

  const maxVal  = Math.max(...timeline, 1);
  const bins    = timeline.length;
  const stepX   = chartW / (bins - 1);
  const durSecs = metadata.duration_seconds || 30;
  const binSecs = durSecs / bins;

  const xOf = i => PAD_L + i * stepX;
  const yOf = v => PAD_T + chartH - (v / maxVal) * chartH;

  // grid lines
  ctx.strokeStyle = 'rgba(212,145,90,.06)';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD_T + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(PAD_L + chartW, y);
    ctx.stroke();
  }

  // filled area
  const grad = ctx.createLinearGradient(0, PAD_T, 0, PAD_T + chartH);
  grad.addColorStop(0,   'rgba(212,145,90,.22)');
  grad.addColorStop(1,   'rgba(212,145,90,.01)');

  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(timeline[0]));
  for (let i = 1; i < bins; i++) {
    const cx = xOf(i - 1) + stepX * 0.5;
    ctx.bezierCurveTo(cx, yOf(timeline[i - 1]), cx, yOf(timeline[i]), xOf(i), yOf(timeline[i]));
  }
  ctx.lineTo(xOf(bins - 1), PAD_T + chartH);
  ctx.lineTo(xOf(0), PAD_T + chartH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // line
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(timeline[0]));
  for (let i = 1; i < bins; i++) {
    const cx = xOf(i - 1) + stepX * 0.5;
    ctx.bezierCurveTo(cx, yOf(timeline[i - 1]), cx, yOf(timeline[i]), xOf(i), yOf(timeline[i]));
  }
  ctx.strokeStyle = '#D4915A';
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // dots at data points
  ctx.fillStyle = '#D4915A';
  for (let i = 0; i < bins; i++) {
    ctx.beginPath();
    ctx.arc(xOf(i), yOf(timeline[i]), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // x-axis labels (show 5 evenly-spaced)
  ctx.fillStyle   = 'rgba(107,94,78,.8)';
  ctx.font        = `500 9px 'JetBrains Mono', monospace`;
  ctx.textAlign   = 'center';
  ctx.textBaseline = 'top';
  const labelStep = Math.max(1, Math.floor(bins / 5));
  for (let i = 0; i < bins; i += labelStep) {
    const tSec = Math.round(i * binSecs);
    const lbl  = tSec >= 60
      ? `${Math.floor(tSec/60)}m${tSec%60 ? String(tSec%60).padStart(2,'0')+'s' : ''}`
      : `${tSec}s`;
    ctx.fillText(lbl, xOf(i), PAD_T + chartH + 6);
  }

  // y-axis max label
  ctx.textAlign   = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(maxVal), PAD_L - 4, PAD_T);
}

// ── donut chart (Canvas 2D) ────────────────────────────────

function drawDonutChart(vehicleComposition) {
  const canvas = document.getElementById('donut-chart');
  const legend = document.getElementById('donut-legend');
  if (!canvas) return;

  const composition = vehicleComposition || {};
  const entries = Object.entries(composition).filter(([, v]) => v > 0);
  if (!entries.length) return;

  const dpr    = window.devicePixelRatio || 1;
  const size   = Math.min(canvas.offsetWidth, 180);
  const h      = 180;

  canvas.width  = canvas.offsetWidth * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const cx     = canvas.offsetWidth / 2;
  const cy     = h / 2;
  const outer  = Math.min(cx, cy) - 4;
  const inner  = outer * 0.58;
  const total  = entries.reduce((s, [, v]) => s + v, 0);

  let angle = -Math.PI / 2;

  entries.forEach(([key, value]) => {
    const slice = (value / total) * Math.PI * 2;
    const color = DONUT_COLORS[key] || '#555';

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outer, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    angle += slice;
  });

  // donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, Math.PI * 2);
  ctx.fillStyle = '#141310';
  ctx.fill();

  // center label
  ctx.fillStyle    = 'rgba(237,229,216,.7)';
  ctx.font         = `500 11px 'JetBrains Mono', monospace`;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(total), cx, cy - 7);
  ctx.fillStyle = 'rgba(107,94,78,.8)';
  ctx.font      = `400 8px 'JetBrains Mono', monospace`;
  ctx.fillText('vehicles', cx, cy + 7);

  // build legend
  if (legend) {
    legend.innerHTML = '';
    entries.forEach(([key, value]) => {
      const pct  = Math.round((value / total) * 100);
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `
        <span class="legend-dot" style="background:${DONUT_COLORS[key] || '#555'}"></span>
        <span>${key} <span style="color:var(--text2)">${pct}%</span></span>
      `;
      legend.appendChild(item);
    });
  }
}

// ── confidence bars ────────────────────────────────────────

function animateConfBars(summary) {
  const { detection_confidence, tracking_accuracy, classification_precision } = summary;

  const fmt = v => (v != null ? `${Number(v).toFixed(1)}%` : '—');

  setText('conf-detection', fmt(detection_confidence));
  setText('conf-tracking',  fmt(tracking_accuracy));
  setText('conf-class',     fmt(classification_precision));

  // delay so CSS transition is visible
  setTimeout(() => {
    const setBar = (barId, pbarId, value) => {
      const bar  = document.getElementById(barId);
      const pbar = document.getElementById(pbarId);
      if (!bar) return;
      const pct = Math.min(100, Math.max(0, value || 0));
      bar.style.width = pct + '%';
      if (pbar) {
        pbar.setAttribute('aria-valuenow', Math.round(pct));
      }
    };
    setBar('conf-bar-detection', 'conf-bar-detection', detection_confidence);
    setBar('conf-bar-tracking',  'conf-bar-tracking',  tracking_accuracy);
    setBar('conf-bar-class',     'conf-bar-class',      classification_precision);
  }, 400);
}

// ── resize handling ────────────────────────────────────────

function initResizeHandler(summary, metadata) {
  let raf;
  const handler = () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      drawFlowChart(summary, metadata);
      drawDonutChart(summary.vehicle_composition);
    });
  };
  window.addEventListener('resize', handler);
}

// ── boot ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const videoId  = localStorage.getItem('tl_video_id')  || 'demo_001';
    const filename = localStorage.getItem('tl_filename')  || '';

    populateNav(filename);
    initModalControls();

    let data;
    try {
      data = await loadAnalysis(videoId);
    } catch (fetchErr) {
      console.error('ARGUS: failed to load analysis', fetchErr);
      showError(
        'SESSION EXPIRED — analysis results were not found.<br>' +
        'This can happen if the server restarted or the session timed out.'
      );
      return;
    }

    if (!data || !data.metadata || !data.summary) {
      showError('INVALID DATA — the analysis response was malformed.');
      return;
    }

    const { summary, metadata, incidents = [] } = data;

    populateStats(data);

    buildIncidentLog(incidents, videoId, (incident) => {
      openModal(incident, videoId);
    });

    // charts — run after layout so offsetWidth is correct
    requestAnimationFrame(() => {
      drawFlowChart(summary, metadata);
      drawDonutChart(summary.vehicle_composition);
      animateConfBars(summary);
      initResizeHandler(summary, metadata);
    });

  } catch (err) {
    console.error('ARGUS dashboard: unhandled error', err);
    showError(
      `UNEXPECTED ERROR — ${err.message || 'unknown error'}.<br>` +
      'Open the browser console for details.'
    );
  }
});
