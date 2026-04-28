import { describe, it, expect } from 'vitest';
import { estimateHr } from '../../src/core/hr/estimate';
import { syntheticTrace } from '../helpers/synthetic';
import { detrend } from '../../src/core/dsp/detrend';
import { bandpassBiquad, filtfilt } from '../../src/core/dsp/butterworth';
import { mean } from '../../src/core/dsp/normalize';

describe('estimateHr', () => {
  it('recovers BPM of a clean tone', () => {
    const fps = 30, N = fps * 10, bpm = 72;
    const f = bpm / 60;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * f * i / fps);
    const r = estimateHr(x, fps);
    expect(Math.abs(r.bpm - bpm)).toBeLessThan(1);
    expect(r.snr).toBeGreaterThan(5);
    expect(r.confidence).toBeGreaterThan(0.9);
  });

  it('low confidence on white noise', () => {
    const fps = 30, N = fps * 10;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.random() - 0.5;
    const r = estimateHr(x, fps);
    expect(r.confidence).toBeLessThan(0.5);
  });

  it('recovers BPM through moderate noise (SNR=0 dB on green channel)', () => {
    const fps = 30, durationSec = 15, bpm = 84;
    const trace = syntheticTrace({ bpm, fps, durationSec, snrDb: 0 });
    // Detrend + bandpass green channel as Phase 4 methods will do.
    const m = mean(trace.g);
    const centered = new Float32Array(trace.g.length);
    for (let i = 0; i < centered.length; i++) centered[i] = trace.g[i] - m;
    let pulse = detrend(centered, 100);
    const { b, a } = bandpassBiquad(2, 0.7, 4, fps);
    pulse = filtfilt(b, a, pulse);
    const r = estimateHr(pulse, fps);
    expect(Math.abs(r.bpm - bpm)).toBeLessThan(2);
    expect(r.confidence).toBeGreaterThan(0.3);
  });
});
