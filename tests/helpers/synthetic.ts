import type { RgbTrace } from '../../src/core/types';

const PBV_CANONICAL: [number, number, number] = [0.33, 0.78, 0.53];

// Deterministic PRNG so synthetic test cases don't flake on RNG variance.
// mulberry32 — fast, good distribution for test fixtures, NOT cryptographically secure.
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate an RGB trace with a known BPM injected into all 3 channels via PBV ratios,
 * plus per-channel additive Gaussian noise.
 *
 * `snrDb` is the SNR (10*log10(signalPower/noisePower)) on the GREEN channel — the
 * dominant cardiac channel under canonical PBV ratios. Red and blue channels have
 * proportionally lower SNR (by 20*log10(pbv[i]/pbv[1]) dB).
 *
 * `seed` (optional) controls the deterministic PRNG used to draw the additive
 * Gaussian noise. If omitted, the seed is derived from
 * `bpm * 1000 + round(fps) * 100 + round(snrDb * 10)` so each (bpm, fps, snrDb)
 * combination is reproducible across runs but different combinations get
 * different noise realizations across the test grid.
 */
export function syntheticTrace(opts: {
  bpm: number;
  fps: number;
  durationSec: number;
  snrDb: number;
  pbv?: [number, number, number];
  seed?: number;
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

  const defaultSeed =
    opts.bpm * 1000 + Math.round(opts.fps) * 100 + Math.round(opts.snrDb * 10);
  const rng = mulberry32(opts.seed ?? defaultSeed);

  for (let n = 0; n < N; n++) {
    const t = n / opts.fps;
    const carrier = Math.sin(2 * Math.PI * f * t);
    r[n] = dc + ac * pbv[0] * carrier + noiseStd * gaussian(rng);
    g[n] = dc + ac * pbv[1] * carrier + noiseStd * gaussian(rng);
    b[n] = dc + ac * pbv[2] * carrier + noiseStd * gaussian(rng);
  }
  return { r, g, b, fps: opts.fps };
}

function gaussian(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
