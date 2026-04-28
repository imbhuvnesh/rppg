export type IIR = { b: Float64Array; a: Float64Array };

/**
 * RBJ-cookbook constant-peak-gain bandpass biquad. center frequency
 * f0 = sqrt(low*high), Q = f0 / (high - low). order=1 is a single biquad;
 * order=2 cascades the biquad with itself (sharper biquad — not a true
 * 4th-order Butterworth). Returns IIR coefficients normalized by a0.
 */
export function bandpassBiquad(order: 1 | 2, lowHz: number, highHz: number, fs: number): IIR {
  const f0 = Math.sqrt(lowHz * highHz);
  const Q = f0 / (highHz - lowHz);
  const w0 = 2 * Math.PI * f0 / fs;
  const alpha = Math.sin(w0) / (2 * Q);
  const cosw0 = Math.cos(w0);
  // RBJ bandpass (constant peak gain)
  const b0 = alpha;
  const b1 = 0;
  const b2 = -alpha;
  const a0 = 1 + alpha;
  const a1 = -2 * cosw0;
  const a2 = 1 - alpha;
  const b = new Float64Array([b0 / a0, b1 / a0, b2 / a0]);
  const a = new Float64Array([1, a1 / a0, a2 / a0]);
  if (order === 1) return { b, a };
  // order 2: cascade — convolve coefficient polynomials
  const bb = convolve(b, b);
  const aa = convolve(a, a);
  return { b: bb, a: aa };
}

function convolve(x: Float64Array, y: Float64Array): Float64Array {
  const out = new Float64Array(x.length + y.length - 1);
  for (let i = 0; i < x.length; i++)
    for (let j = 0; j < y.length; j++) out[i + j] += x[i] * y[j];
  return out;
}

/**
 * Direct-form transposed IIR filter. Returns y same length as x. State is
 * initialized to zero; expect ~order samples of startup transient.
 */
export function lfilter(b: Float64Array, a: Float64Array, x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    let s = 0;
    for (let i = 0; i < b.length; i++) if (n - i >= 0) s += b[i] * x[n - i];
    for (let i = 1; i < a.length; i++) if (n - i >= 0) s -= a[i] * y[n - i];
    y[n] = s / a[0];
  }
  return y;
}

/**
 * Zero-phase forward-backward filter. Effective magnitude response is
 * |H(f)|^2. Pads input with odd reflection at both edges to suppress edge
 * transients. Returns y same length as x.
 */
export function filtfilt(b: Float64Array, a: Float64Array, x: Float32Array): Float32Array {
  if (x.length === 0) return new Float32Array(0);
  // Pad reflection at edges to reduce transient.
  const pad = Math.min(3 * Math.max(b.length, a.length), x.length - 1);
  const N = x.length;
  const ext = new Float32Array(N + 2 * pad);
  for (let i = 0; i < pad; i++) ext[i] = 2 * x[0] - x[pad - i];
  ext.set(x, pad);
  for (let i = 0; i < pad; i++) ext[N + pad + i] = 2 * x[N - 1] - x[N - 2 - i];
  const f = lfilter(b, a, ext);
  const r = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) r[i] = f[f.length - 1 - i];
  const fr = lfilter(b, a, r);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = fr[fr.length - 1 - pad - i];
  return out;
}
