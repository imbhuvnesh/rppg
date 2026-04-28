import { LiveRppg } from '../src/browser';
import type { MethodName, RppgResult } from '../src/core/types';

const video = document.querySelector<HTMLVideoElement>('#video')!;
const bpmEl = document.querySelector<HTMLDivElement>('#bpm')!;
const wave = document.querySelector<HTMLCanvasElement>('#wave')!;
const overlay = document.querySelector<HTMLCanvasElement>('#overlay')!;
const methodSel = document.querySelector<HTMLSelectElement>('#method')!;
const status = document.querySelector<HTMLDivElement>('#status')!;
const snrBar = document.querySelector<HTMLDivElement>('#snr-bar')!;

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
const FOREHEAD = [10, 109, 67, 103, 54, 21, 162, 127, 234];
const LEFT_CHEEK = [50, 101, 36, 205, 187, 123, 116];
const RIGHT_CHEEK = [280, 330, 266, 425, 411, 352, 345];

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

function handleResult(r: RppgResult) {
  if (r.confidence < 0.3) {
    status.textContent = 'Low confidence';
    bpmEl.textContent = '--';
    bpmEl.style.opacity = '0.4';
  } else {
    status.textContent = '';
    bpmEl.textContent = r.bpm.toFixed(0);
    bpmEl.style.opacity = '1';
  }
  drawWaveform(wave, r.pulseSignal);
  snrBar.style.width = Math.max(0, Math.min(100, r.snr * 10)) + '%';
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
