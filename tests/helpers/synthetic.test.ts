import { describe, it, expect } from 'vitest';
import { syntheticTrace } from './synthetic';
import { fftMagnitude, nextPow2 } from '../../src/core/dsp/fft';

describe('syntheticTrace', () => {
  it('green channel peak frequency matches injected BPM', () => {
    const fps = 30;
    const bpm = 90;
    const durationSec = 10;
    const trace = syntheticTrace({ bpm, fps, durationSec, snrDb: 30 });

    expect(trace.fps).toBe(fps);
    expect(trace.r.length).toBe(fps * durationSec);
    expect(trace.g.length).toBe(fps * durationSec);
    expect(trace.b.length).toBe(fps * durationSec);

    // FFT the green channel; expect the spectral peak in the HR band to be
    // near 1.5 Hz (= 90 BPM). We center the signal first to drop the DC bin.
    const g = trace.g;
    const mean = g.reduce((s, v) => s + v, 0) / g.length;
    const centered = new Float32Array(g.length);
    for (let i = 0; i < g.length; i++) centered[i] = g[i] - mean;

    const N = nextPow2(centered.length);
    const mag = fftMagnitude(centered);
    const binHz = fps / N;
    const lo = Math.max(1, Math.floor(0.7 / binHz));
    const hi = Math.min(N / 2 - 1, Math.ceil(4.0 / binHz));
    let peak = lo;
    for (let k = lo; k <= hi; k++) if (mag[k] > mag[peak]) peak = k;
    const peakHz = peak * binHz;
    const expectedHz = bpm / 60;

    expect(Math.abs(peakHz - expectedHz)).toBeLessThan(0.1);

    // Sanity-check the PBV mapping: spectral magnitude at the heart-rate peak
    // should be largest in green (pbv = 0.78) and smallest in red (pbv = 0.33).
    // This catches an accidental r/g/b swap in the helper.
    const peakAmp = (ch: Float32Array): number => {
      const m = ch.reduce((s, v) => s + v, 0) / ch.length;
      const c = new Float32Array(ch.length);
      for (let i = 0; i < ch.length; i++) c[i] = ch[i] - m;
      const M = fftMagnitude(c);
      return M[peak];
    };
    const ar = peakAmp(trace.r);
    const ag = peakAmp(trace.g);
    const ab = peakAmp(trace.b);
    expect(ag).toBeGreaterThan(ab);
    expect(ab).toBeGreaterThan(ar);
  });
});
