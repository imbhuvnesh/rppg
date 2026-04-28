export type IIR = { b: Float64Array; a: Float64Array };

// Order is per low/high side; bandpass = order*2 final order.
// We support order = 1 (final 2) and order = 2 (final 4).
// Implementation: design analog prototype, frequency-transform to bandpass, bilinear -> digital.
// For simplicity here we implement orders 1 and 2 via the cookbook biquad bandpass with center
// frequency f0 = sqrt(low*high) and Q = f0 / (high - low), cascaded for order 2.
export function butterBandpass(order: 1 | 2, lowHz: number, highHz: number, fs: number): IIR {
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

export function filtfilt(b: Float64Array, a: Float64Array, x: Float32Array): Float32Array {
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
