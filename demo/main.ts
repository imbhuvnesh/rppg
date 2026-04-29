import { LiveRppg } from '../src/browser';
import type { MethodName, RppgResult } from '../src/core/types';

const video = document.querySelector<HTMLVideoElement>('#video')!;
const bpmEl = document.querySelector<HTMLDivElement>('#bpm')!;
const wave = document.querySelector<HTMLCanvasElement>('#wave')!;
const overlay = document.querySelector<HTMLCanvasElement>('#overlay')!;
const methodSel = document.querySelector<HTMLSelectElement>('#method')!;
const status = document.querySelector<HTMLDivElement>('#status')!;
const snrBar = document.querySelector<HTMLDivElement>('#snr-bar')!;
const diag = document.querySelector<HTMLDivElement>('#diag')!;

let live: LiveRppg | null = null;

async function main() {
  status.textContent = 'Requesting camera...';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: 640, height: 480 },
  });
  video.srcObject = stream;
  await video.play();
  resizeOverlayToVideo();
  status.textContent = 'Calibrating...';
  start();
  requestAnimationFrame(drawOverlay);
}

function resizeOverlayToVideo() {
  // Match overlay backing-store size to the video resolution so coordinates
  // line up with normalized landmarks scaled by videoWidth/videoHeight.
  if (video.videoWidth && video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
}

// FaceMesh landmark indices for the ROI polygons drawn on the overlay.
// Kept in sync with FaceRoiTracker (forehead + cheeks) so the overlay
// reflects exactly where the rPPG signal is sampled from.
const FOREHEAD = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251];
const LEFT_CHEEK = [117, 118, 119, 120, 100, 101, 50];
const RIGHT_CHEEK = [346, 347, 348, 349, 329, 330, 280];

const isDebug = new URLSearchParams(location.search).has('debug');

function drawPoly(
  ctx: CanvasRenderingContext2D,
  indices: number[],
  landmarks: { x: number; y: number }[],
  W: number,
  H: number
) {
  ctx.beginPath();
  for (let i = 0; i < indices.length; i++) {
    const lm = landmarks[indices[i]];
    if (!lm) continue;
    const x = lm.x * W;
    const y = lm.y * H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawRoiPolygons(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number }[],
  W: number,
  H: number
) {
  ctx.strokeStyle = 'rgba(74, 222, 128, 0.7)';
  ctx.lineWidth = 2;
  drawPoly(ctx, FOREHEAD, landmarks, W, H);
  drawPoly(ctx, LEFT_CHEEK, landmarks, W, H);
  drawPoly(ctx, RIGHT_CHEEK, landmarks, W, H);
}

function drawDebugLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number }[],
  W: number,
  H: number
) {
  ctx.save();
  ctx.font = '8px monospace';
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (!lm) continue;
    const x = lm.x * W;
    const y = lm.y * H;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.beginPath();
    ctx.arc(x, y, 1, 0, Math.PI * 2);
    ctx.fill();
    if (i % 10 === 0) {
      ctx.fillStyle = 'rgba(255,255,0,0.9)';
      ctx.fillText(String(i), x + 2, y - 2);
    }
  }
  ctx.restore();
}

function drawOverlay() {
  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const roi = live?.getRoi();
  if (roi && roi.landmarks.length > 0) {
    if (video.videoWidth && overlay.width !== video.videoWidth) {
      overlay.width = video.videoWidth;
    }
    if (video.videoHeight && overlay.height !== video.videoHeight) {
      overlay.height = video.videoHeight;
    }
    drawRoiPolygons(ctx, roi.landmarks, overlay.width, overlay.height);
    if (isDebug) drawDebugLandmarks(ctx, roi.landmarks, overlay.width, overlay.height);
    // Face is back — clear our own "Looking for face..." latch so handleResult
    // can take over status text on the next pipeline tick.
    if (status.textContent === 'Looking for face...') {
      status.textContent = '';
    }
  } else {
    // No face detected — only set this status if not currently showing a more specific message.
    if (status.textContent === '' || status.textContent === 'Calibrating...') {
      status.textContent = 'Looking for face...';
    }
  }
  requestAnimationFrame(drawOverlay);
}

function start() {
  live?.stop();
  live = new LiveRppg({
    method: methodSel.value as MethodName,
    windowSec: 10,
    updateHz: 1,
  });
  live.onUpdate(handleResult);
  live.start(video).catch((err: unknown) => {
    status.textContent = 'Error: ' + errMsg(err);
  });
}

// Status strings owned by handleResult. The rAF overlay loop sets
// "Looking for face..." / "Calibrating..." which we must not overwrite —
// we only mutate status when the current text is one of these (an empty
// string, or a previous handleResult-owned label).
const HANDLED_STATUS = new Set([
  '',
  'Looking for face...',
  'Calibrating...',
  'Low confidence',
  '(weak signal)',
  '(low confidence)',
]);

function handleResult(r: RppgResult) {
  const inRange = Number.isFinite(r.bpm) && r.bpm >= 40 && r.bpm <= 200;

  if (!inRange) {
    bpmEl.textContent = '--';
    bpmEl.style.opacity = '0.4';
    if (HANDLED_STATUS.has(status.textContent ?? '')) {
      status.textContent = 'Low confidence';
    }
  } else {
    bpmEl.textContent = r.bpm.toFixed(0);
    if (r.confidence >= 0.5) {
      bpmEl.style.opacity = '1';
      if (HANDLED_STATUS.has(status.textContent ?? '')) {
        status.textContent = '';
      }
    } else if (r.confidence >= 0.2) {
      bpmEl.style.opacity = '0.7';
      if (HANDLED_STATUS.has(status.textContent ?? '')) {
        status.textContent = '(weak signal)';
      }
    } else {
      bpmEl.style.opacity = '0.4';
      if (HANDLED_STATUS.has(status.textContent ?? '')) {
        status.textContent = '(low confidence)';
      }
    }
  }

  drawWaveform(wave, r.pulseSignal);
  snrBar.style.width = Math.max(0, Math.min(100, r.snr * 10)) + '%';

  const fps = live?.observedFps() ?? 0;
  const bufSec = live?.bufferSeconds() ?? 0;
  updateDiag(methodSel.value, r, fps, bufSec);
}

function updateDiag(method: string, r: RppgResult, observedFps: number, bufferSec: number) {
  const bpmStr = Number.isFinite(r.bpm) ? r.bpm.toFixed(1) : 'NaN';
  diag.textContent = [
    `method: ${method}    fps: ${observedFps.toFixed(1)}`,
    `buffer: ${bufferSec.toFixed(1)} s   bpm: ${bpmStr}`,
    `snr: ${r.snr.toFixed(1)} dB    conf: ${r.confidence.toFixed(2)}`,
  ].join('\n');
}

function drawWaveform(canvas: HTMLCanvasElement, x: Float32Array) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (x.length === 0) return;
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 2;
  ctx.beginPath();
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < x.length; i++) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
  }
  const range = (max - min) || 1;
  for (let i = 0; i < x.length; i++) {
    const px = (i / (x.length - 1)) * W;
    const py = H - ((x[i] - min) / range) * H;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function errMsg(err: unknown): string {
  if (err && typeof err === 'object' && 'message' in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
}

methodSel.addEventListener('change', start);
video.addEventListener('loadedmetadata', resizeOverlayToVideo);
main().catch((err: unknown) => {
  status.textContent = 'Init error: ' + errMsg(err);
});
