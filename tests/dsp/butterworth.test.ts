import { describe, it, expect } from 'vitest';
import { bandpassBiquad, filtfilt } from '../../src/core/dsp/butterworth';

const sine = (f: number, fs: number, N: number) => {
  const x = new Float32Array(N);
  for (let n = 0; n < N; n++) x[n] = Math.sin(2 * Math.PI * f * n / fs);
  return x;
};
const rms = (x: Float32Array) => {
  let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
};

describe('butterworth', () => {
  it('passes in-band tones with < 1 dB attenuation', () => {
    const fs = 30, N = 2000;
    const { b, a } = bandpassBiquad(2, 0.7, 4, fs);
    const x = sine(1.5, fs, N);
    const y = filtfilt(b, a, x);
    // skip transients
    const ratio = rms(y.subarray(200, N - 200)) / rms(x.subarray(200, N - 200));
    const dB = 20 * Math.log10(ratio);
    expect(dB).toBeGreaterThan(-1);
  });

  it('rejects out-of-band tones by >= 30 dB', () => {
    const fs = 30, N = 4000;
    const { b, a } = bandpassBiquad(2, 0.7, 4, fs);
    const x = sine(10, fs, N);
    const y = filtfilt(b, a, x);
    const ratio = rms(y.subarray(400, N - 400)) / rms(x.subarray(400, N - 400));
    const dB = 20 * Math.log10(ratio);
    expect(dB).toBeLessThan(-30);
  });

  it('filtfilt is zero-phase: peak position unchanged for an impulse', () => {
    const fs = 30, N = 1024;
    const { b, a } = bandpassBiquad(2, 0.7, 4, fs);
    const x = new Float32Array(N);
    x[N / 2] = 1; // impulse at center
    const y = filtfilt(b, a, x);
    let peak = 0;
    for (let i = 1; i < N; i++) if (Math.abs(y[i]) > Math.abs(y[peak])) peak = i;
    expect(Math.abs(peak - N / 2)).toBeLessThan(2);
  });
});
