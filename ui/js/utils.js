// ═══════════════════════════════════════════════════════════
// ARGUS AI — utils.js
// Shared helper functions used across all pages
// ═══════════════════════════════════════════════════════════

/**
 * Format seconds → "01:22"
 */
export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/**
 * Format seconds → "3m 42s"
 */
export function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s}s`;
}

/**
 * Format km/h value → "38 km/h"
 */
export function formatSpeed(kmh) {
  return `${Math.round(kmh)} km/h`;
}

/**
 * Format 0–1 float or 0–100 float → "94.2%"
 */
export function formatConfidence(value) {
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(1)}%`;
}

/**
 * Format bytes → "124.3 MB"
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

/**
 * Format seconds offset → "T+01:22"
 */
export function formatTimestamp(seconds) {
  return `T+${formatTime(seconds)}`;
}

/**
 * Debounce a function call
 */
export function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Linear interpolation
 */
export function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * Clamp value between min and max
 */
export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Generate a short session identifier → "0x4A2F"
 */
export function generateSessionId() {
  const hex = Math.floor(Math.random() * 0xffff)
    .toString(16)
    .toUpperCase()
    .padStart(4, '0');
  return `0x${hex}`;
}

/**
 * Map severity string to hex color
 * high → #ff4d6d  medium → #ff9f43  low → #4a9eff
 */
export function severityColor(severity) {
  switch ((severity || '').toLowerCase()) {
    case 'high':   return '#ff4d6d';
    case 'medium': return '#ff9f43';
    case 'low':    return '#4a9eff';
    default:       return '#4a9eff';
  }
}

/**
 * Map causal factor type to human-readable label
 */
export function causalLabel(type) {
  const map = {
    tailgating:           'Tailgating',
    unsafe_lane_change:   'Unsafe Lane Change',
    failure_to_yield:     'Failure to Yield',
    speed_differential:   'Speed Differential',
    obstruction:          'Road Obstruction',
  };
  return map[(type || '').toLowerCase()] || type;
}

/**
 * Map vehicle class to a single display character
 */
export function vehicleClassIcon(cls) {
  switch ((cls || '').toLowerCase()) {
    case 'car':        return '◆';
    case 'motorcycle': return '▲';
    case 'truck':      return '■';
    case 'bus':        return '●';
    default:           return '○';
  }
}

/**
 * Resize a canvas to match its parent's width, keep given height.
 * Returns the 2D context.
 * @param {HTMLCanvasElement} canvas
 * @param {number} [height] — explicit pixel height, defaults to current height
 */
export function resizeCanvas(canvas, height) {
  const parent = canvas.parentElement;
  const w = parent ? parent.offsetWidth : canvas.offsetWidth;
  const h = height !== undefined ? height : canvas.offsetHeight;
  const dpr = window.devicePixelRatio || 1;

  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width  = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

/**
 * Parse simple markdown-ish text into HTML
 * Handles **bold**, *italic*, `code`, \n\n → paragraphs
 */
export function parseMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code class="inline-code">$1</code>')
    .replace(/\n\n/g,           '</p><p>')
    .replace(/^/,               '<p>')
    .replace(/$/,               '</p>');
}
