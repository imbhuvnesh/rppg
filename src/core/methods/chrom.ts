import type { RgbTrace, RppgMethod } from '../types';
import { mean, std } from '../dsp/normalize';
import { hann } from '../dsp/hann';
import { bandpassBiquad, filtfilt } from '../dsp/butterworth';

/**
 * CHROM method (de Haan & Jeanne 2013).
 *
 * Sliding Hann-windowed overlap-add. Per window of length W = ceil(1.6 * fps)
 * with 50% overlap:
 *   - mean-normalize R, G, B (Xn = X / mean(X) per channel)
 *   - Xs = 3*Rn - 2*Gn
 *   - Ys = 1.5*Rn + Gn - 1.5*Bn
 *   - alpha = std(Xs) / std(Ys)
 *   - pulse_window = Xs - alpha * Ys
 *   - apply Hann window, accumulate into output via overlap-add.
 *
 * The Xs/Ys bandpass is applied to the whole-trace Xs/Ys arrays once (before
 * windowing) rather than inside each tiny 1.6 s window. Per-window filtfilt
 * suffers from transient edge effects on a 1.6 s buffer at 60 fps, which
 * destroys low-BPM recovery.
 *
 * Note: per-window `alpha = std(Xw)/std(Yw)` is computed on the
 * bandpassed Xs/Ys, so it differs slightly from canonical-CHROM `alpha`
 * (which uses the unfiltered windows). rPPG-Toolbox `CHROME_DEHAAN`
 * takes the same approach; tests confirm BPM recovery within tolerance.
 *
 * Inner Xs/Ys band: 0.5-2.5 Hz, order 1. The inner filter's job is baseline-
 * drift suppression (drift is typically <0.1 Hz from breathing/lighting/motion),
 * not HR range gating — that's the outer pipeline filter. A lower edge of
 * 0.7 Hz puts 50 BPM (0.833 Hz) right at the band edge and creates a deep
 * confidence dip with the RBJ-cookbook biquad's response shape; 0.5 Hz still
 * kills drift while keeping the test grid clear of the dip.
 */
export const chrom: RppgMethod = (trace: RgbTrace): Float32Array => {
  const { r, g, b, fps } = trace;
  const N = r.length;
  const W = Math.ceil(1.6 * fps);
  const stride = Math.max(1, Math.floor(W / 2));
  const out = new Float32Array(N);

  if (N < W) return out;

  const hannW = hann(W);
  // Order 1, 0.5-2.5 Hz: drift suppression only; HR-band gating is the outer
  // pipeline filter's job. Order 2 + 0.7 Hz lower edge created a deep band-edge
  // dip that killed confidence at BPM=50 / fps=60.
  const filt = bandpassBiquad(1, 0.5, 2.5, fps);

  // Whole-trace mean-normalize R, G, B and form Xs, Ys.
  const mrAll = mean(r) || 1;
  const mgAll = mean(g) || 1;
  const mbAll = mean(b) || 1;
  const Xs = new Float32Array(N);
  const Ys = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const rn = r[i] / mrAll;
    const gn = g[i] / mgAll;
    const bn = b[i] / mbAll;
    Xs[i] = 3 * rn - 2 * gn;
    Ys[i] = 1.5 * rn + gn - 1.5 * bn;
  }
  // Bandpass Xs and Ys at 0.5-2.5 Hz on the whole trace (zero-phase filtfilt).
  const XsF = filtfilt(filt.b, filt.a, Xs);
  const YsF = filtfilt(filt.b, filt.a, Ys);

  // Sliding Hann-windowed overlap-add. alpha is computed per window.
  for (let start = 0; start + W <= N; start += stride) {
    const Xw = XsF.subarray(start, start + W);
    const Yw = YsF.subarray(start, start + W);
    const sx = std(Xw);
    const sy = std(Yw) || 1;
    const alpha = sx / sy;
    for (let i = 0; i < W; i++) {
      out[start + i] += (Xw[i] - alpha * Yw[i]) * hannW[i];
    }
  }
  return out;
};
