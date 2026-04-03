// ═══════════════════════════════════════════════════════════
// ARGUS AI — incident.js
// Single incident detail page logic
// ═══════════════════════════════════════════════════════════

import { getIncident, getAnalysis, streamClaudeAnalysis } from './api.js';
import {
  formatTimestamp, formatTime, formatConfidence,
  causalLabel, severityColor, resizeCanvas, generateSessionId, debounce,
} from './utils.js';

// ── State ─────────────────────────────────────────────────────

let incident      = null;
let metadata      = null;
let summary       = null;
let trajectories  = [];
let activePersona = localStorage.getItem('tl_persona') || 'insurance';

// ── Boot ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  const params     = new URLSearchParams(window.location.search);
  const incidentId = params.get('id') || '3f7a1c2d-8e4b-4a9f-b6c2-1d0e5f3a7b8c';

  try {
    const result = await getIncident(incidentId);
    incident     = result.incident;
    metadata     = result.metadata;
    summary      = result.summary;
    trajectories = result.trajectories || [];
  } catch (e) {
    console.error('[ARGUS] Incident load error:', e);
    document.body.innerHTML += `
      <div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
           background:var(--bg-deep);">
        <div style="background:var(--bg-card);border:1px solid var(--border);
             border-left:2px solid var(--red);padding:28px 32px;text-align:center;max-width:400px;">
          <span style="font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:3px;
               color:var(--red);display:block;margin-bottom:12px;text-transform:uppercase;">
            Error Loading Incident
          </span>
          <span style="font-family:'Barlow',sans-serif;font-weight:300;font-size:12px;
               color:rgba(255,255,255,0.45);">${e.message}</span>
          <a href="dashboard.html" style="display:block;margin-top:16px;"
             class="btn-ghost">← Back to Dashboard</a>
        </div>
      </div>
    `;
    return;
  }

  populateNav(incidentId);
  populateBreadcrumb(incidentId);
  populateHeader(incident, metadata);
  populateTimelineBar(incident, metadata);
  populateVideoPlayer(incident, metadata);
  populateCausalList(incident);
  populateTrajectoryCanvas(incident, trajectories, metadata);
  populateEventTimeline(incident);
  initAIPanel(incident);
  initIntersectionObserver();
  updatePersonaUI(activePersona);

  // Single resize listener
  window.addEventListener('resize', debounce(() => populateTrajectoryCanvas(incident, trajectories, metadata), 150));

  // Animate causal bars after small delay
  setTimeout(() => animateCausalBars(incident), 300);
});

// ── Nav & breadcrumb ──────────────────────────────────────────

function populateNav(id) {
  const el = document.getElementById('nav-breadcrumb');
  if (el) el.textContent = `ARGUS / Dashboard / ${id.toUpperCase()}`;
}

function populateBreadcrumb(id) {
  const el = document.getElementById('breadcrumb-id');
  if (el) el.textContent = `Incident ${id.toUpperCase()}`;
}

// ── Header ────────────────────────────────────────────────────

function populateHeader(inc, meta) {
  // Badges
  const badgesEl = document.getElementById('inc-badges');
  if (badgesEl) {
    badgesEl.innerHTML = `
      <span class="badge badge-${inc.severity}">${inc.severity.toUpperCase()}</span>
      <span class="badge badge-info">${inc.type.replace(/_/g, ' ').toUpperCase()}</span>
    `;
  }

  // Title
  const titleEl = document.getElementById('inc-title');
  if (titleEl) {
    titleEl.textContent = inc.type === 'accident'
      ? 'Vehicle Accident Detection'
      : inc.type === 'risky_interaction'
        ? 'Risky Vehicle Interaction'
        : 'Vehicle Near-Miss Event';
  }

  // Timestamp
  const tsEl = document.getElementById('inc-timestamp');
  if (tsEl) {
    tsEl.textContent = `${formatTimestamp(inc.timestamp_start)} — ${formatTimestamp(inc.timestamp_end)}`;
  }

  // Vehicle pills
  const pillsEl = document.getElementById('vehicle-pills');
  if (pillsEl) {
    if (inc.vehicles_involved.length > 0) {
      pillsEl.innerHTML = inc.vehicles_involved
        .map(v => {
          const label = typeof v === 'number' ? `V${String(v).padStart(2, '0')}` : v;
          return `<span class="vehicle-pill">${label}</span>`;
        })
        .join('');
    } else {
      pillsEl.innerHTML = `<span style="font-family:'Barlow',sans-serif;font-weight:300;font-size:11px;color:rgba(255,255,255,0.2);">No specific vehicles identified</span>`;
    }
  }

  // Confidence
  const confEl = document.getElementById('inc-conf-value');
  if (confEl) confEl.textContent = `${Math.round(inc.confidence * 100)}%`;

  // Mini causal bars (header right)
  const miniEl = document.getElementById('causal-mini');
  if (miniEl) {
    miniEl.innerHTML = inc.causal_factors.map(cf => `
      <div class="causal-mini-row">
        <div class="causal-mini-label">
          <span>${causalLabel(cf.type)}</span>
          <span>${Math.round(cf.confidence * 100)}%</span>
        </div>
        <div class="causal-mini-bar-bg">
          <div class="causal-mini-bar-fill" style="width:0%"
               data-target="${cf.confidence * 100}%"></div>
        </div>
      </div>
    `).join('');

    setTimeout(() => {
      miniEl.querySelectorAll('.causal-mini-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.target;
      });
    }, 300);
  }

  // AI panel incident ID
  const aiIdEl = document.getElementById('ai-inc-id');
  if (aiIdEl) aiIdEl.textContent = inc.id.toUpperCase();
}

// ── Timeline bar ──────────────────────────────────────────────

function populateTimelineBar(inc, meta) {
  const dur      = meta.duration_seconds;
  const startPct = (inc.timestamp_start / dur) * 100;
  const endPct   = (inc.timestamp_end   / dur) * 100;
  const widthPct = endPct - startPct;
  const color    = severityColor(inc.severity);

  const region = document.getElementById('timeline-incident-region');
  if (region) {
    region.style.cssText = `
      left:${startPct}%; width:${widthPct}%;
      background:${color}22; border-left:2px solid ${color};
      position:absolute; top:0; height:100%;
    `;
  }

  const cursor = document.getElementById('timeline-cursor');
  if (cursor) cursor.style.left = `${startPct}%`;

  const safeSet = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  safeSet('tbar-inc-start', formatTimestamp(inc.timestamp_start));
  safeSet('tbar-inc-end',   formatTimestamp(inc.timestamp_end));
  safeSet('tbar-duration',  formatTimestamp(dur));

  // Note: timeline-bar click is handled by populateVideoPlayer (seeks video + drives cursor)
}

// ── Video player ──────────────────────────────────────────────

function populateVideoPlayer(inc, meta) {
  const video    = document.getElementById('incident-video');
  const fallback = document.getElementById('video-fallback');
  if (!video) return;

  const src = meta && meta.video_url ? meta.video_url : null;
  if (!src) {
    video.style.display   = 'none';
    if (fallback) fallback.style.display = 'flex';
    return;
  }

  video.src = src;

  // Seek to incident start once metadata is loaded
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = inc.timestamp_start;
  }, { once: true });

  // If video fails to load, show fallback
  video.addEventListener('error', () => {
    video.style.display = 'none';
    if (fallback) fallback.style.display = 'flex';
  }, { once: true });

  // Sync timeline cursor with video playback
  const cursor = document.getElementById('timeline-cursor');
  if (cursor && meta) {
    video.addEventListener('timeupdate', () => {
      const pct = (video.currentTime / meta.duration_seconds) * 100;
      cursor.style.left = `${Math.min(pct, 100)}%`;
    });
  }

  // Timeline bar click seeks video
  const bar = document.getElementById('timeline-bar');
  if (bar && meta) {
    bar.addEventListener('click', (e) => {
      const rect = bar.getBoundingClientRect();
      const pct  = (e.clientX - rect.left) / rect.width;
      video.currentTime = pct * meta.duration_seconds;
    });
  }
}

// ── Causal list ───────────────────────────────────────────────

function populateCausalList(inc) {
  const el = document.getElementById('causal-list');
  if (!el) return;

  el.innerHTML = inc.causal_factors.map(cf => `
    <div class="causal-item">
      <div class="causal-item-header">
        <span class="causal-item-label">${causalLabel(cf.type)}</span>
        <span class="causal-item-pct">${Math.round(cf.confidence * 100)}%</span>
      </div>
      <div class="causal-bar-bg">
        <div class="causal-bar-fill" style="width:0%" data-target="${cf.confidence * 100}%"></div>
      </div>
      <div class="causal-desc">${cf.description || ''}</div>
    </div>
  `).join('');
}

function animateCausalBars(inc) {
  document.querySelectorAll('.causal-bar-fill, .causal-mini-bar-fill').forEach(bar => {
    if (bar.dataset.target) bar.style.width = bar.dataset.target;
  });
}

// ── Trajectory canvas ─────────────────────────────────────────

const TRAJ_COLORS = ['#4a9eff', '#00e5ff', '#ff9f43', '#a78bfa', '#ff4d6d', '#4ade80'];

function populateTrajectoryCanvas(inc, trajs, meta) {
  const canvas = document.getElementById('trajectory-canvas');
  if (!canvas) return;

  const H   = 180;
  const ctx = resizeCanvas(canvas, H);
  const W   = canvas.offsetWidth;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#020408';
  ctx.fillRect(0, 0, W, H);

  // Parse video resolution for coordinate scaling
  const [videoW, videoH] = meta && meta.resolution
    ? meta.resolution.split('x').map(Number)
    : [1280, 720];
  const scaleX = W / videoW;
  const scaleY = H / videoH;

  // Grid lines (every 25% of video height)
  ctx.strokeStyle = 'rgba(74,158,255,0.07)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([8, 6]);
  for (let r = 1; r < 4; r++) {
    const y = (r / 4) * H;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Find trajectories that match the incident's involved vehicles
  const involvedIds = inc.vehicles_involved.map(v =>
    typeof v === 'number' ? v : parseInt(String(v).replace(/\D/g, ''), 10)
  );

  // Build matched list — fall back to drawing mock arcs if none found
  const matched = (trajs || []).filter(t => involvedIds.includes(t.vehicle_id));

  if (matched.length === 0) {
    drawMockArcs(ctx, W, H, inc, involvedIds);
    return;
  }

  // Incident window in ms
  const incStartMs = inc.timestamp_start * 1000;
  const incEndMs   = inc.timestamp_end   * 1000;

  matched.forEach((traj, i) => {
    const color  = TRAJ_COLORS[i % TRAJ_COLORS.length];
    const frames = traj.frames;
    if (!frames || frames.length < 2) return;

    // Draw full path faded
    ctx.beginPath();
    frames.forEach((f, fi) => {
      const x = f.center[0] * scaleX;
      const y = f.center[1] * scaleY;
      fi === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = `${color}44`;
    ctx.lineWidth   = 1;
    ctx.stroke();

    // Draw incident-window segment highlighted
    const incFrames = frames.filter(f => f.timestamp_ms >= incStartMs - 100 && f.timestamp_ms <= incEndMs + 100);
    if (incFrames.length >= 2) {
      ctx.beginPath();
      incFrames.forEach((f, fi) => {
        const x = f.center[0] * scaleX;
        const y = f.center[1] * scaleY;
        fi === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    // Speed-colored dots every N frames
    frames.forEach((f, fi) => {
      if (fi % 3 !== 0) return;
      const x     = f.center[0] * scaleX;
      const y     = f.center[1] * scaleY;
      const spd   = f.speed_estimate || 0;
      const t     = Math.min(spd / 120, 1);
      const dotColor = speedColor(t);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
    });

    // Start dot
    const first = frames[0];
    ctx.beginPath();
    ctx.arc(first.center[0] * scaleX, first.center[1] * scaleY, 4, 0, Math.PI * 2);
    ctx.fillStyle = `${color}88`;
    ctx.fill();

    // Vehicle ID label at last frame
    const last = frames[frames.length - 1];
    const lx   = last.center[0] * scaleX;
    const ly   = last.center[1] * scaleY;
    ctx.fillStyle = color;
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'center';
    ctx.fillText(`V${String(traj.vehicle_id).padStart(2, '0')}`, lx, Math.max(ly - 8, 10));

    // Arrow at last frame
    if (frames.length >= 2) {
      const prev  = frames[frames.length - 2];
      const dx    = last.center[0] - prev.center[0];
      const dy    = last.center[1] - prev.center[1];
      const angle = Math.atan2(dy * scaleY, dx * scaleX);
      const aLen  = 7;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(lx - aLen * Math.cos(angle - 0.4), ly - aLen * Math.sin(angle - 0.4));
      ctx.moveTo(lx, ly);
      ctx.lineTo(lx - aLen * Math.cos(angle + 0.4), ly - aLen * Math.sin(angle + 0.4));
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1;
      ctx.stroke();
    }
  });

  // Legend: speed color scale
  drawSpeedLegend(ctx, W, H);
}

/** Map speed 0→1 to a blue→green→red gradient */
function speedColor(t) {
  if (t < 0.5) {
    const r = Math.round(74  + (255 - 74)  * (t * 2));
    const g = Math.round(158 + (159 - 158) * (t * 2));
    const b = Math.round(255 + (67  - 255) * (t * 2));
    return `rgb(${r},${g},${b})`;
  } else {
    const r = Math.round(255);
    const g = Math.round(159 + (79  - 159) * ((t - 0.5) * 2));
    const b = Math.round(67  + (67  - 67)  * ((t - 0.5) * 2));
    return `rgb(${r},${g},${b})`;
  }
}

function drawSpeedLegend(ctx, W, H) {
  const bW = 60;
  const bH = 4;
  const x  = W - bW - 8;
  const y  = H - 16;
  const grad = ctx.createLinearGradient(x, 0, x + bW, 0);
  grad.addColorStop(0,   '#4a9eff');
  grad.addColorStop(0.5, '#ff9f43');
  grad.addColorStop(1,   '#ff4d6d');
  ctx.fillStyle = grad;
  ctx.fillRect(x, y, bW, bH);
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font      = '6px "Share Tech Mono"';
  ctx.textAlign = 'left';
  ctx.fillText('0', x, y + bH + 8);
  ctx.textAlign = 'right';
  ctx.fillText('120 km/h', x + bW, y + bH + 8);
}

/** Fallback mock arcs when no real trajectory data matches */
function drawMockArcs(ctx, W, H, inc, involvedIds) {
  const lanes  = 4;
  const laneH  = H / lanes;
  const labels = involvedIds.length > 0
    ? involvedIds.map(v => `V${String(v).padStart(2, '0')}`)
    : ['V??'];

  labels.forEach((label, i) => {
    const color     = TRAJ_COLORS[i % TRAJ_COLORS.length];
    const startLane = i % lanes;
    const endLane   = (startLane + (i === 0 ? 1 : -1) + lanes) % lanes;
    const sx = W * 0.12, ex = W * 0.88;
    const sy = startLane * laneH + laneH * 0.5;
    const ey = endLane   * laneH + laneH * 0.5;
    const cpY = (sy + ey) / 2 - 20;

    const grad = ctx.createLinearGradient(sx, 0, ex, 0);
    grad.addColorStop(0, `${color}33`);
    grad.addColorStop(0.5, color);
    grad.addColorStop(1, `${color}55`);

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(W * 0.5, cpY, ex, ey);
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    const angle = Math.atan2(ey - cpY, ex - W * 0.5);
    const aLen  = 8;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - aLen * Math.cos(angle - 0.4), ey - aLen * Math.sin(angle - 0.4));
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - aLen * Math.cos(angle + 0.4), ey - aLen * Math.sin(angle + 0.4));
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1;
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font      = '7px "Share Tech Mono"';
    ctx.textAlign = 'center';
    ctx.fillText(label, ex - 8, ey - 10);

    ctx.beginPath();
    ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = `${color}88`;
    ctx.fill();
  });
}

// ── Event timeline list ───────────────────────────────────────

function populateEventTimeline(inc) {
  const el = document.getElementById('timeline-list');
  if (!el) return;

  const color = severityColor(inc.severity);

  const events = [
    { ts: inc.timestamp_start,                           desc: 'Incident window begins — initial detection triggered' },
    { ts: (inc.timestamp_start + inc.timestamp_end) / 2, desc: inc.description },
    { ts: inc.timestamp_end,                             desc: 'Incident window ends — vehicles returned to normal flow' },
  ];

  // Add causal factor events
  inc.causal_factors.forEach((cf, i) => {
    const ts = inc.timestamp_start + (i + 0.5) * ((inc.timestamp_end - inc.timestamp_start) / (inc.causal_factors.length + 1));
    events.splice(1 + i, 0, {
      ts,
      desc: `${causalLabel(cf.type)} detected — ${Math.round(cf.confidence * 100)}% confidence`,
    });
  });

  events.sort((a, b) => a.ts - b.ts);

  el.innerHTML = events.map(evt => `
    <div class="timeline-item">
      <div class="timeline-dot" style="background:${color};"></div>
      <div>
        <div class="timeline-ts">${formatTimestamp(evt.ts)}</div>
        <div class="timeline-desc">${evt.desc}</div>
      </div>
    </div>
  `).join('');
}

// ── AI panel ──────────────────────────────────────────────────

const PERSONA_LABELS = {
  insurance:  'Insurance Investigator',
  engineer:   'Traffic Engineer',
  researcher: 'Road Safety Researcher',
};

function updatePersonaUI(persona) {
  document.querySelectorAll('.persona-mini-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.persona === persona);
  });
  const badge = document.getElementById('ai-inc-persona-badge');
  if (badge) badge.textContent = PERSONA_LABELS[persona] || persona;
}

function typewriterEffect(element, text, speed = 16) {
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

function initAIPanel(inc) {
  // Persona tabs
  document.querySelectorAll('.persona-mini-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      activePersona = tab.dataset.persona;
      localStorage.setItem('tl_persona', activePersona);
      updatePersonaUI(activePersona);
      const area = document.getElementById('ai-incident-text');
      if (area) {
        area.classList.remove('streaming');
        area.textContent = 'Select persona and generate incident analysis.';
      }
    });
  });

  // Analyse button
  const btn  = document.getElementById('btn-analyse');
  const area = document.getElementById('ai-incident-text');
  if (!btn || !area) return;

  btn.addEventListener('click', async () => {
    btn.disabled    = true;
    btn.textContent = 'Analysing…';
    area.classList.remove('streaming');
    area.textContent = '';

    // Build incident-specific analysis data payload
    const incidentPayload = {
      metadata,
      summary,
      incident: inc,
    };

    const chunks  = [];
    let streamDone = false;

    try {
      await new Promise((resolve, reject) => {
        streamClaudeAnalysis(incidentPayload, activePersona,
          (chunk) => chunks.push(chunk),
          () => { streamDone = true; resolve(); }
        );
        setTimeout(() => { if (!streamDone) resolve(); }, 5000);
      });
    } catch (e) {
      console.warn('[ARGUS] Claude stream error:', e);
    }

    const fullText = chunks.join('') || buildMockIncidentAnalysis(inc, activePersona);
    if (fullText.includes('backend not connected') || fullText.includes('unavailable')) {
      await typewriterEffect(area, buildMockIncidentAnalysis(inc, activePersona), 16);
    } else {
      await typewriterEffect(area, fullText, 16);
    }

    btn.disabled    = false;
    btn.textContent = 'Analyse Incident';
  });
}

function buildMockIncidentAnalysis(inc, persona) {
  const conf  = Math.round(inc.confidence * 100);
  const ts    = formatTimestamp(inc.timestamp_start);
  const vList = inc.vehicles_involved
    .map(v => typeof v === 'number' ? `V${String(v).padStart(2, '0')}` : v)
    .join(', ') || 'unidentified vehicles';
  const cf    = inc.causal_factors.map(c => causalLabel(c.type)).join(' and ');

  const templates = {
    insurance: `Incident ${inc.id.toUpperCase()} at ${ts} presents a ${inc.severity.toLowerCase()}-severity ${inc.type.toLowerCase().replace('_', ' ')} event with ${conf}% model confidence. The causal analysis identifies ${cf} as primary contributing factors. ${vList !== 'unidentified vehicles' ? `Vehicles ${vList} are directly implicated` : 'Vehicle identification was inconclusive'} — ${inc.description.toLowerCase()}. For liability assessment, this event warrants ${conf >= 85 ? 'immediate prioritisation given the high confidence score' : 'further manual review before drawing fault conclusions'}.`,
    engineer: `From a traffic engineering perspective, incident ${inc.id.toUpperCase()} at ${ts} reflects ${cf.toLowerCase()} conditions at this intersection. The ${inc.severity.toLowerCase()} severity classification with ${conf}% detection confidence suggests a systemic flow issue rather than an isolated event. Infrastructure response options include adaptive signal timing adjustment and geometric review of the conflict zone. ${inc.vehicles_involved.length > 1 ? `Multi-vehicle involvement (${vList}) indicates a capacity or sight-distance deficiency.` : ''}`,
    researcher: `Incident ${inc.id.toUpperCase()} (${formatTimestamp(inc.timestamp_start)}–${formatTimestamp(inc.timestamp_end)}) provides a ${conf}% confidence data point for ${inc.type.replace('_', ' ').toLowerCase()} research classification. Primary causal factors ${cf} align with established near-miss taxonomy. ${inc.vehicles_involved.length > 0 ? `Vehicle cohort size (n=${inc.vehicles_involved.length}) is consistent with documented interaction patterns at signalised intersections.` : ''} Confidence score of ${conf}% falls ${conf >= 80 ? 'within acceptable bounds' : 'below the 80% threshold'} for inclusion in statistical analyses without manual verification.`,
  };

  return templates[persona] || templates.insurance;
}

// ── IntersectionObserver ──────────────────────────────────────

function initIntersectionObserver() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.fade-in-section').forEach(el => observer.observe(el));
}
