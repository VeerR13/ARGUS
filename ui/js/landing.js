// ═══════════════════════════════════════════════════════════
// ARGUS AI — landing.js
// Three.js Earth scene + upload flow with progress polling
// ═══════════════════════════════════════════════════════════

import { uploadVideo, pollJobStatus } from './api.js';
import { formatFileSize } from './utils.js';

// ── Three.js scene ───────────────────────────────────────────

let renderer, scene, camera, animFrameId;
let earthMesh, cloudMesh, starPoints;
let cityNodes    = [];
let particles    = [];
let arcPointSets = [];   // array of arrays of THREE.Vector3

const CITY_PAIRS = [
  [[51.5,-0.1],   [48.9,2.3]],     // London ↔ Paris
  [[51.5,-0.1],   [40.7,-74.0]],   // London ↔ New York
  [[51.5,-0.1],   [25.2,55.3]],    // London ↔ Dubai
  [[48.9,2.3],    [52.5,13.4]],    // Paris ↔ Berlin
  [[48.9,2.3],    [30.1,31.2]],    // Paris ↔ Cairo
  [[52.5,13.4],   [55.7,37.6]],    // Berlin ↔ Moscow
  [[55.7,37.6],   [39.9,116.4]],   // Moscow ↔ Beijing
  [[39.9,116.4],  [35.7,139.7]],   // Beijing ↔ Tokyo
  [[39.9,116.4],  [1.3,103.8]],    // Beijing ↔ Singapore
  [[35.7,139.7],  [34.0,-118.2]],  // Tokyo ↔ LA
  [[35.7,139.7],  [-33.9,151.2]],  // Tokyo ↔ Sydney
  [[34.0,-118.2], [40.7,-74.0]],   // LA ↔ New York
  [[34.0,-118.2], [-23.5,-46.6]],  // LA ↔ São Paulo
  [[40.7,-74.0],  [-23.5,-46.6]],  // New York ↔ São Paulo
  [[40.7,-74.0],  [41.9,-87.6]],   // New York ↔ Chicago
  [[-23.5,-46.6], [6.5,3.4]],      // São Paulo ↔ Lagos
  [[6.5,3.4],     [30.1,31.2]],    // Lagos ↔ Cairo
  [[30.1,31.2],   [25.2,55.3]],    // Cairo ↔ Dubai
  [[25.2,55.3],   [19.1,72.9]],    // Dubai ↔ Mumbai
  [[19.1,72.9],   [1.3,103.8]],    // Mumbai ↔ Singapore
  [[1.3,103.8],   [-33.9,151.2]],  // Singapore ↔ Sydney
];

function latLonToVec3(lat, lon, r) {
  const phi   = (90 - lat) * Math.PI / 180;
  const theta = (lon + 180) * Math.PI / 180;
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
     r * Math.cos(phi),
     r * Math.sin(phi) * Math.sin(theta)
  );
}

function greatCirclePoints(lat1, lon1, lat2, lon2, n, r) {
  const v1 = latLonToVec3(lat1, lon1, r).normalize();
  const v2 = latLonToVec3(lat2, lon2, r).normalize();
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const v = new THREE.Vector3().copy(v1).lerp(v2, t).normalize().multiplyScalar(r);
    pts.push(v);
  }
  return pts;
}

function initScene() {
  const canvas = document.getElementById('earth-canvas');

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x020408, 1);
  renderer.setSize(window.innerWidth, window.innerHeight);

  // Scene
  scene = new THREE.Scene();

  // Camera
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 1.6, 3.0);
  camera.lookAt(0, 0.2, 0);

  // Lighting
  const sun = new THREE.DirectionalLight(0xfff5e0, 1.2);
  sun.position.set(2, 1.5, 1);
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0x8ab4ff, 0.07));

  const loader = new THREE.TextureLoader();

  // Earth
  const earthGeo = new THREE.SphereGeometry(1, 64, 64);
  const earthMat = new THREE.MeshPhongMaterial({
    map:         loader.load('https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg'),
    specularMap: loader.load('https://threejs.org/examples/textures/planets/earth_specular_2048.jpg'),
    normalMap:   loader.load('https://threejs.org/examples/textures/planets/earth_normal_2048.jpg'),
    shininess:   18,
  });
  earthMesh = new THREE.Mesh(earthGeo, earthMat);
  scene.add(earthMesh);

  // Clouds
  const cloudGeo = new THREE.SphereGeometry(1.005, 64, 64);
  const cloudMat = new THREE.MeshPhongMaterial({
    map:         loader.load('https://threejs.org/examples/textures/planets/earth_clouds_1024.png'),
    transparent: true,
    opacity:     0.38,
    depthWrite:  false,
  });
  cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
  scene.add(cloudMesh);

  // Atmospheric rim glow
  const atmGeo = new THREE.SphereGeometry(1.02, 64, 64);
  const atmMat = new THREE.MeshBasicMaterial({
    color:       0x4488ff,
    transparent: true,
    opacity:     0.10,
    side:        THREE.BackSide,
  });
  scene.add(new THREE.Mesh(atmGeo, atmMat));

  // Stars
  const starGeo = new THREE.BufferGeometry();
  const starPos = [];
  for (let i = 0; i < 600; i++) {
    // Scatter in upper-left space region
    starPos.push(
      (Math.random() - 0.8) * 12,
      (Math.random() - 0.1) * 8,
      (Math.random() - 0.9) * 10
    );
  }
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.008, sizeAttenuation: true });
  starPoints = new THREE.Points(starGeo, starMat);
  scene.add(starPoints);

  // Road arcs + city nodes
  const uniqueCities = new Map(); // key "lat,lon" → {lat,lon}

  CITY_PAIRS.forEach(([[lat1, lon1], [lat2, lon2]]) => {
    const pts = greatCirclePoints(lat1, lon1, lat2, lon2, 48, 1.001);
    arcPointSets.push(pts);

    // Primary arc line
    const geo1 = new THREE.BufferGeometry().setFromPoints(pts);
    const mat1 = new THREE.LineBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.55 });
    scene.add(new THREE.Line(geo1, mat1));

    // Glow arc line
    const geo2 = new THREE.BufferGeometry().setFromPoints(pts);
    const mat2 = new THREE.LineBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.15 });
    scene.add(new THREE.Line(geo2, mat2));

    uniqueCities.set(`${lat1},${lon1}`, { lat: lat1, lon: lon1 });
    uniqueCities.set(`${lat2},${lon2}`, { lat: lat2, lon: lon2 });
  });

  // City node dots
  uniqueCities.forEach(({ lat, lon }) => {
    const pos = latLonToVec3(lat, lon, 1.001);
    const geo = new THREE.SphereGeometry(0.006, 6, 6);
    const mat = new THREE.MeshBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.8 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    mesh.userData.phase = Math.random() * Math.PI * 2;
    scene.add(mesh);
    cityNodes.push(mesh);
  });

  // Traffic particles
  for (let i = 0; i < 18; i++) {
    const arcIdx = Math.floor(Math.random() * arcPointSets.length);

    // Head
    const headGeo = new THREE.SphereGeometry(0.004, 6, 6);
    const headMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const head    = new THREE.Mesh(headGeo, headMat);
    scene.add(head);

    // Ghost 1
    const g1Geo = new THREE.SphereGeometry(0.003, 6, 6);
    const g1Mat = new THREE.MeshBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.55 });
    const g1    = new THREE.Mesh(g1Geo, g1Mat);
    scene.add(g1);

    // Ghost 2
    const g2Geo = new THREE.SphereGeometry(0.002, 6, 6);
    const g2Mat = new THREE.MeshBasicMaterial({ color: 0x1a3a6a, transparent: true, opacity: 0.22 });
    const g2    = new THREE.Mesh(g2Geo, g2Mat);
    scene.add(g2);

    particles.push({
      head, g1, g2,
      arcIdx,
      progress: Math.random(),
      speed: 0.0008 + Math.random() * 0.0014,
    });
  }
}

function getArcPosition(arcIdx, progress) {
  const pts = arcPointSets[arcIdx];
  if (!pts || pts.length === 0) return new THREE.Vector3();
  const t   = Math.max(0, Math.min(1, progress));
  const raw = t * (pts.length - 1);
  const lo  = Math.floor(raw);
  const hi  = Math.min(lo + 1, pts.length - 1);
  const frac = raw - lo;
  return new THREE.Vector3().lerpVectors(pts[lo], pts[hi], frac);
}

let clock = 0;

function animate() {
  animFrameId = requestAnimationFrame(animate);
  clock += 0.016;

  // Rotate Earth & clouds
  if (earthMesh) earthMesh.rotation.y  += 0.00009;
  if (cloudMesh) cloudMesh.rotation.y  += 0.000045;

  // City node pulse
  cityNodes.forEach(node => {
    const s = 0.8 + 0.4 * (0.5 + 0.5 * Math.sin(clock * 1.5 + node.userData.phase));
    node.scale.setScalar(s);
  });

  // Traffic particles
  particles.forEach(p => {
    p.progress += p.speed;
    if (p.progress > 1) {
      p.progress = 0;
      p.arcIdx   = Math.floor(Math.random() * arcPointSets.length);
    }

    const pos  = getArcPosition(p.arcIdx, p.progress);
    const pos1 = getArcPosition(p.arcIdx, Math.max(0, p.progress - 0.015));
    const pos2 = getArcPosition(p.arcIdx, Math.max(0, p.progress - 0.030));

    p.head.position.copy(pos);
    p.g1.position.copy(pos1);
    p.g2.position.copy(pos2);
  });

  renderer.render(scene, camera);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// ── HUD coordinate ticker ────────────────────────────────────

let altCounter  = 420;
let hudTick     = 0;
let latBase     = 48.8566;
let lonBase     = 2.3522;

const HUD_STATUSES = [
  'SCANNING', 'ACQUIRING', 'TRACKING', 'ANALYSING', 'LOCKED ON', 'MONITORING',
];

function initHUD() {
  const hudCoords = document.getElementById('hud-coords');
  if (!hudCoords) return;

  setInterval(() => {
    hudTick++;
    altCounter = altCounter <= 0 ? 420 : altCounter - 1;

    // Drift lat/lon slightly to feel live
    const latDrift = latBase + Math.sin(hudTick * 0.031) * 0.0012;
    const lonDrift = lonBase + Math.cos(hudTick * 0.019) * 0.0008;

    const alt    = String(altCounter).padStart(4, '0');
    const lat    = latDrift.toFixed(4);
    const lon    = lonDrift.toFixed(4);
    const status = HUD_STATUSES[Math.floor(hudTick / 18) % HUD_STATUSES.length];

    hudCoords.innerHTML = `
      LAT ${lat}° N<br>
      LON &nbsp;${lon}° E<br>
      ALT &nbsp;${alt} KM<br>
      SYS &nbsp;${status}<span class="hud-cursor"></span>
    `;
  }, 90);
}

// ── Upload flow ──────────────────────────────────────────────

const ACCEPTED_TYPES = new Set([
  'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm',
  'video/x-matroska', 'image/jpeg', 'image/png',
]);

function getProgressElements() {
  return {
    wrap:    document.getElementById('upload-progress-wrap'),
    fill:    document.getElementById('upload-progress-fill'),
    status:  document.getElementById('upload-progress-status'),
    pct:     document.getElementById('upload-progress-pct'),
  };
}

function showProgress() {
  const { wrap } = getProgressElements();
  if (wrap) wrap.classList.add('visible');
}

function updateProgress(pct, message) {
  const { fill, status, pct: pctEl } = getProgressElements();
  if (fill)   fill.style.width = `${pct}%`;
  if (status) status.textContent = message;
  if (pctEl)  pctEl.textContent  = `${pct}%`;
}

function showLoadingOverlay() {
  const overlay = document.getElementById('loading-overlay');
  if (!overlay) return;
  overlay.classList.add('visible');

  const fill = document.getElementById('loading-bar-fill');
  if (fill) {
    fill.style.width = '0%';
    requestAnimationFrame(() => fill.classList.add('animate'));
  }
}

function cycleLoadingMessages() {
  const el = document.getElementById('loading-status-text');
  if (!el) return;

  const msgs = [
    'Initializing neural pipeline…',
    'Extracting frame sequences…',
    'Running object detection…',
    'Computing optical flow…',
    'Classifying vehicle types…',
    'Aggregating trajectory data…',
    'Generating research report…',
  ];

  let i = 0;
  el.textContent = msgs[0];
  return setInterval(() => {
    i = (i + 1) % msgs.length;
    el.textContent = msgs[i];
  }, 370);
}

async function handleFile(file) {
  if (!file) return;

  // Validate type
  if (!ACCEPTED_TYPES.has(file.type) && !file.name.match(/\.(mp4|mov|avi|webm|mkv|jpg|png)$/i)) {
    showError('Unsupported file type. Please use MP4, MOV, AVI, WEBM, or MKV.');
    return;
  }

  // Persist state
  localStorage.setItem('tl_filename', file.name);
  localStorage.setItem('tl_filesize', file.size);

  // Disable upload zone
  const uploadBox = document.getElementById('upload-box');
  const selectBtn = document.getElementById('select-btn');
  if (uploadBox) uploadBox.style.pointerEvents = 'none';
  if (selectBtn) { selectBtn.disabled = true; selectBtn.textContent = 'Processing…'; }

  // Show progress bar below the upload box
  showProgress();
  updateProgress(0, 'Uploading file…');

  try {
    // Upload
    const { video_id } = await uploadVideo(file);
    localStorage.setItem('tl_video_id', video_id);

    // Show full-screen loading overlay
    showLoadingOverlay();
    const msgInterval = cycleLoadingMessages();

    // Poll progress
    pollJobStatus(
      video_id,
      (pct, msg) => {
        updateProgress(pct, msg);
        // Sync loading bar with actual progress
        const fill = document.getElementById('loading-bar-fill');
        if (fill) fill.style.width = `${pct}%`;
        const statusEl = document.getElementById('loading-status-text');
        if (statusEl) statusEl.textContent = msg;
      },
      (analysisData) => {
        clearInterval(msgInterval);
        // Persist full analysis for dashboard
        try {
          localStorage.setItem('tl_analysis', JSON.stringify(analysisData));
        } catch (e) {
          // localStorage size limit — skip caching large JSON
          console.warn('[ARGUS] Could not cache analysis data:', e.message);
        }
        // Fade out overlay, then navigate
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
          overlay.style.transition = 'opacity 0.4s ease';
          overlay.style.opacity = '0';
        }
        setTimeout(() => {
          window.location.href = `dashboard.html?id=${video_id}`;
        }, 450);
      },
      (err) => {
        clearInterval(msgInterval);
        console.error('[ARGUS] Poll error:', err);
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.classList.remove('visible');
        showError(`Processing failed: ${err.message}`);
        // Re-enable upload
        if (uploadBox) uploadBox.style.pointerEvents = '';
        if (selectBtn) { selectBtn.disabled = false; selectBtn.textContent = '— SELECT FILE FOR ANALYSIS —'; }
      }
    );
  } catch (err) {
    console.error('[ARGUS] Upload error:', err);
    showError(`Upload failed: ${err.message}`);
    if (uploadBox) uploadBox.style.pointerEvents = '';
    if (selectBtn) { selectBtn.disabled = false; selectBtn.textContent = '— SELECT FILE FOR ANALYSIS —'; }
  }
}

function showError(msg) {
  let el = document.getElementById('upload-error');
  if (!el) {
    el = document.createElement('div');
    el.id = 'upload-error';
    el.style.cssText = `
      margin-top: 12px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 8px;
      letter-spacing: 2px;
      color: #ff4d6d;
      text-transform: uppercase;
      padding: 8px 12px;
      border: 1px solid rgba(255,77,109,0.3);
      background: rgba(255,77,109,0.06);
    `;
    document.getElementById('upload-zone')?.appendChild(el);
  }
  el.textContent = msg;
}

// ── Init ─────────────────────────────────────────────────────

function initUpload() {
  const uploadBox  = document.getElementById('upload-box');
  const selectBtn  = document.getElementById('select-btn');
  const fileInput  = document.getElementById('file-input');

  if (!uploadBox || !selectBtn || !fileInput) return;

  // Click to open file picker
  uploadBox.addEventListener('click', () => fileInput.click());
  selectBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

  // Drag events
  uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
  });

  uploadBox.addEventListener('dragleave', (e) => {
    if (!uploadBox.contains(e.relatedTarget)) {
      uploadBox.classList.remove('dragover');
    }
  });

  uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });

  // File input change
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
    fileInput.value = '';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initScene();
  animate();
  initHUD();
  initUpload();
  window.addEventListener('resize', onResize);
});
