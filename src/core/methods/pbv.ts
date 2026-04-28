import type { RgbTrace } from '../types';
import { mean, std } from '../dsp/normalize';

const PBV_CANONICAL: [number, number, number] = [0.33, 0.78, 0.53];

export type PbvOptions = {
  /**
   * If true, estimate the blood-volume signature from the trace itself as
   * `[std(R), std(G), std(B)]` (then normalised to unit length). If false (default),
   * use the canonical PBV signature `[0.33, 0.78, 0.53]` (de Haan & van Leest 2014).
   */
  estimatePbvFromTrace?: boolean;
};

/**
 * PBV method (de Haan & van Leest 2014).
 *
 * 1. Mean-normalize each channel: x / mean(x).
 * 2. Subtract per-channel mean from the normalized signal.
 * 3. Build C as a 3xN matrix.
 * 4. PBV signature defaults to canonical [0.33, 0.78, 0.53] (unit-normalized).
 *    If `estimatePbvFromTrace` is true, the signature is [std(R), std(G), std(B)]
 *    of the centered/normalized signal, unit-normalized.
 * 5. Compute Q = C * C^T (3x3).
 * 6. Solve W = Q^-1 * pbv (closed-form 3x3 inverse).
 * 7. pulse = W^T * C  (length N).
 */
export function pbv(trace: RgbTrace, opts: PbvOptions = {}): Float32Array {
  const { r, g, b } = trace;
  const N = r.length;
  if (N < 3) return new Float32Array(N);

  // Mean-normalize per channel (x / mean), then subtract each channel's mean.
  const C = [r, g, b].map(x => {
    const m = mean(x) || 1;
    const norm = new Float32Array(N);
    for (let i = 0; i < N; i++) norm[i] = x[i] / m;
    let mNorm = 0;
    for (let i = 0; i < N; i++) mNorm += norm[i];
    mNorm /= N;
    for (let i = 0; i < N; i++) norm[i] -= mNorm;
    return norm;
  });

  // PBV signature.
  let pbvSig: [number, number, number];
  if (opts.estimatePbvFromTrace) {
    pbvSig = [std(C[0]), std(C[1]), std(C[2])];
  } else {
    pbvSig = [...PBV_CANONICAL];
  }
  const pbvNorm = Math.sqrt(pbvSig[0] ** 2 + pbvSig[1] ** 2 + pbvSig[2] ** 2) || 1;
  pbvSig = [pbvSig[0] / pbvNorm, pbvSig[1] / pbvNorm, pbvSig[2] / pbvNorm];

  // Q = C * C^T  (3x3 symmetric).
  const Q: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = i; j < 3; j++) {
    let s = 0;
    for (let n = 0; n < N; n++) s += C[i][n] * C[j][n];
    Q[i][j] = Q[j][i] = s;
  }

  // W = Q^-1 * pbv.
  const Qinv = invert3x3(Q);
  const W: [number, number, number] = [
    Qinv[0][0] * pbvSig[0] + Qinv[0][1] * pbvSig[1] + Qinv[0][2] * pbvSig[2],
    Qinv[1][0] * pbvSig[0] + Qinv[1][1] * pbvSig[1] + Qinv[1][2] * pbvSig[2],
    Qinv[2][0] * pbvSig[0] + Qinv[2][1] * pbvSig[1] + Qinv[2][2] * pbvSig[2],
  ];

  // pulse = W^T * C.
  const pulse = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    pulse[n] = W[0] * C[0][n] + W[1] * C[1][n] + W[2] * C[2][n];
  }
  return pulse;
}

function invert3x3(M: number[][]): number[][] {
  const a = M[0][0], b = M[0][1], c = M[0][2];
  const d = M[1][0], e = M[1][1], f = M[1][2];
  const g = M[2][0], h = M[2][1], i = M[2][2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = det !== 0 ? 1 / det : 0;
  return [
    [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
    [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
    [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
  ];
}
