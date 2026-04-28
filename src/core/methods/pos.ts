import type { RgbTrace, RppgMethod } from '../types';

/**
 * POS method (Wang et al. 2017), locked to rPPG-Toolbox's POS_WANG.
 *
 * Sliding window of length l = ceil(1.6 * fps), stride 1 sample. For each n >= l:
 *   1. Cn = RGB[n-l..n) / mean(RGB[n-l..n)) per channel  (l x 3)
 *   2. S = [[0, 1, -1], [-2, 1, 1]] . Cn^T              (2 x l)
 *   3. h = S[0] + (std(S[0]) / std(S[1])) * S[1]        (length l)
 *   4. h = h - mean(h)
 *   5. H[n-l..n) += h                                   (overlap-add)
 *
 * Returns the raw H buffer; no internal bandpass / detrend (the surrounding
 * pipeline applies those).
 */
export const pos: RppgMethod = (trace: RgbTrace): Float32Array => {
  const { r, g, b, fps } = trace;
  const N = r.length;
  const l = Math.ceil(1.6 * fps);
  const H = new Float32Array(N);
  if (N < l) return H;

  for (let n = l; n <= N; n++) {
    const start = n - l;
    // Compute window means per channel.
    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < l; i++) {
      mr += r[start + i];
      mg += g[start + i];
      mb += b[start + i];
    }
    mr /= l; mg /= l; mb /= l;
    if (mr === 0) mr = 1; if (mg === 0) mg = 1; if (mb === 0) mb = 1;

    // S[0] = G/mg - B/mb, S[1] = -2*R/mr + G/mg + B/mb.
    const s0 = new Float32Array(l);
    const s1 = new Float32Array(l);
    let mean0 = 0, mean1 = 0;
    for (let i = 0; i < l; i++) {
      const rn = r[start + i] / mr;
      const gn = g[start + i] / mg;
      const bn = b[start + i] / mb;
      s0[i] = gn - bn;
      s1[i] = -2 * rn + gn + bn;
      mean0 += s0[i];
      mean1 += s1[i];
    }
    mean0 /= l; mean1 /= l;

    // std of each row.
    let var0 = 0, var1 = 0;
    for (let i = 0; i < l; i++) {
      const d0 = s0[i] - mean0;
      const d1 = s1[i] - mean1;
      var0 += d0 * d0;
      var1 += d1 * d1;
    }
    const sd0 = Math.sqrt(var0 / l);
    const sd1 = Math.sqrt(var1 / l) || 1;
    const ratio = sd0 / sd1;

    // h = S[0] + ratio * S[1], then mean-subtract.
    let mh = 0;
    const h = new Float32Array(l);
    for (let i = 0; i < l; i++) {
      h[i] = s0[i] + ratio * s1[i];
      mh += h[i];
    }
    mh /= l;
    for (let i = 0; i < l; i++) h[i] -= mh;

    // Overlap-add into H.
    for (let i = 0; i < l; i++) H[start + i] += h[i];
  }
  return H;
};
