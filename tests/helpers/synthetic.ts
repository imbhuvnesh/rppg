import type { RgbTrace } from '../../src/core/types';

const PBV_CANONICAL: [number, number, number] = [0.33, 0.78, 0.53];

/**
 * Generate an RGB trace with a known BPM injected into all 3 channels via PBV ratios,
 * plus per-channel additive Gaussian noise.
 *
 * `snrDb` is the SNR (10*log10(signalPower/noisePower)) on the GREEN channel — the
 * dominant cardiac channel under canonical PBV ratios. Red and blue channels have
 * proportionally lower SNR (by 20*log10(pbv[i]/pbv[1]) dB).
 */
export function syntheticTrace(opts: {
  bpm: number;
  fps: number;
  durationSec: number;
  snrDb: number;
  pbv?: [number, number, number];
}): RgbTrace {
  const pbv = opts.pbv ?? PBV_CANONICAL;
  const N = Math.round(opts.fps * opts.durationSec);
  const f = opts.bpm / 60;
  const r = new Float32Array(N), g = new Float32Array(N), b = new Float32Array(N);

  // Skin-reflectance baseline (DC) and pulsatile AC amplitude. Realistic rPPG values
  // (PPG perturbation is ~1-2% of skin reflectance).
  const dc = 0.5;
  const ac = 0.02;

  // Carrier signal power per channel: ((ac * pbv[i])^2) / 2  (sinusoid).
  // Noise power chosen so green-channel SNR = snrDb.
  const greenSigPow = (ac * pbv[1]) ** 2 / 2;
  const noisePow = greenSigPow / Math.pow(10, opts.snrDb / 10);
  const noiseStd = Math.sqrt(noisePow);

  for (let n = 0; n < N; n++) {
    const t = n / opts.fps;
    const carrier = Math.sin(2 * Math.PI * f * t);
    r[n] = dc + ac * pbv[0] * carrier + noiseStd * gaussian();
    g[n] = dc + ac * pbv[1] * carrier + noiseStd * gaussian();
    b[n] = dc + ac * pbv[2] * carrier + noiseStd * gaussian();
  }
  return { r, g, b, fps: opts.fps };
}

function gaussian(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
