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
}

function resizeOverlayToVideo() {
  // Match overlay backing-store size to the video resolution so coordinates
  // line up with normalized landmarks scaled by videoWidth/videoHeight.
  if (video.videoWidth && video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
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
