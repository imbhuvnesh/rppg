import { describe, it, expect } from 'vitest';
import { syntheticTrace } from './synthetic';
import { fftMagnitude, nextPow2 } from '../../src/core/dsp/fft';
import { hann } from '../../src/core/dsp/hann';

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

  it('snrDb param delivers the requested SNR on the green channel', () => {
    const fps = 30;
    const bpm = 90;
    const durationSec = 15;
    const requestedSnrDb = 20;
    const trace = syntheticTrace({ bpm, fps, durationSec, snrDb: requestedSnrDb });

    // Subtract DC from green channel.
    const g = trace.g;
    const m = g.reduce((s, v) => s + v, 0) / g.length;
    const centered = new Float32Array(g.length);
    for (let i = 0; i < g.length; i++) centered[i] = g[i] - m;

    // Apply a Hann window before FFT to suppress sinc-leakage from the (non-bin-aligned)
    // carrier; otherwise carrier sidelobes dominate the "noise" estimate at high SNR.
    // Both signal and noise are equally attenuated by the window, so the SNR ratio is
    // preserved. Then sum |X[k]|^2 in a +/- 0.1 Hz window vs. the rest of the spectrum.
    const w = hann(centered.length);
    const windowed = new Float32Array(centered.length);
    for (let i = 0; i < centered.length; i++) windowed[i] = centered[i] * w[i];

    const N = nextPow2(windowed.length);
    const mag = fftMagnitude(windowed);
    const binHz = fps / N;
    const carrierHz = bpm / 60;
    const carrierBin = Math.round(carrierHz / binHz);
    const tol = Math.max(1, Math.round(0.1 / binHz));

    let totalPow = 0;
    for (let k = 0; k < N; k++) totalPow += mag[k] * mag[k];

    let carrierPow = 0;
    // Sum the positive-frequency window AND its conjugate mirror at N-k.
    for (let k = carrierBin - tol; k <= carrierBin + tol; k++) {
      carrierPow += mag[k] * mag[k];
      carrierPow += mag[N - k] * mag[N - k];
    }

    const noisePow = Math.max(totalPow - carrierPow, 1e-20);
    const measuredSnrDb = 10 * Math.log10(carrierPow / noisePow);

    expect(Math.abs(measuredSnrDb - requestedSnrDb)).toBeLessThan(2);
  });
});
