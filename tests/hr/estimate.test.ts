import { describe, it, expect } from 'vitest';
import { estimateHr } from '../../src/core/hr/estimate';

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
});
