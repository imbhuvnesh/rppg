import { describe, it, expect } from 'vitest';
import { detrend } from '../../src/core/dsp/detrend';

describe('detrend (smoothness-prior)', () => {
  it('removes linear trend from noisy signal', () => {
    const N = 500;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = 0.05 * i + 2 + 0.1 * Math.sin(2 * Math.PI * i / 20) + 0.02 * (Math.random() - 0.5);
    const y = detrend(x, 100);
    let mean = 0;
    for (let i = 0; i < N; i++) mean += y[i];
    mean /= N;
    expect(Math.abs(mean)).toBeLessThan(0.1);
    // residual slope (least squares) should be near zero
    let sxy = 0, sxx = 0, mx = (N - 1) / 2;
    for (let i = 0; i < N; i++) { sxy += (i - mx) * y[i]; sxx += (i - mx) ** 2; }
    const slope = sxy / sxx;
    expect(Math.abs(slope)).toBeLessThan(0.005);
  });

  it('preserves a high-frequency oscillation', () => {
    const N = 500;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * i / 10);
    const y = detrend(x, 100);
    let s1 = 0, s2 = 0;
    for (let i = 50; i < N - 50; i++) { s1 += x[i] * x[i]; s2 += y[i] * y[i]; }
    expect(s2 / s1).toBeGreaterThan(0.8);
  });
});
