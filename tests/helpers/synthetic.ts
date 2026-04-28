// Generate an RGB trace with a known BPM injected into all 3 channels with the
// canonical PBV ratios [0.33, 0.78, 0.53] plus per-channel additive Gaussian noise.

import type { RgbTrace } from '../../src/core/types';

export function syntheticTrace(opts: {
  bpm: number;
  fps: number;
  durationSec: number;
  snrDb: number;
  pbv?: [number, number, number];
}): RgbTrace {
  const pbv = opts.pbv ?? [0.33, 0.78, 0.53];
  const N = Math.round(opts.fps * opts.durationSec);
  const f = opts.bpm / 60;
  const r = new Float32Array(N), g = new Float32Array(N), b = new Float32Array(N);
  const sigPow = 1; // unit-amplitude carrier
  const noisePow = sigPow / Math.pow(10, opts.snrDb / 10);
  const noiseStd = Math.sqrt(noisePow);
  for (let n = 0; n < N; n++) {
    const t = n / opts.fps;
    const carrier = Math.sin(2 * Math.PI * f * t);
    // skin reflection baseline ~ 0.5; perturbation amplitude ~ 0.02 (typical)
    const dc = 0.5;
    const ac = 0.02;
    r[n] = dc + ac * pbv[0] * carrier + noiseStd * 0.02 * gaussian();
    g[n] = dc + ac * pbv[1] * carrier + noiseStd * 0.02 * gaussian();
    b[n] = dc + ac * pbv[2] * carrier + noiseStd * 0.02 * gaussian();
  }
  return { r, g, b, fps: opts.fps };
}

function gaussian() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
