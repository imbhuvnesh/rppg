import { describe, it, expect } from 'vitest';
import { fftMagnitude, nextPow2 } from '../../src/core/dsp/fft';

describe('fft', () => {
  it('finds the bin of a pure tone', () => {
    const N = 1024;
    const fs = 100;
    const f = 5; // Hz
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = Math.sin(2 * Math.PI * f * n / fs);
    const mag = fftMagnitude(x);
    let peakIdx = 0;
    for (let i = 1; i < mag.length / 2; i++) if (mag[i] > mag[peakIdx]) peakIdx = i;
    const peakHz = peakIdx * fs / N;
    expect(Math.abs(peakHz - f)).toBeLessThan(0.2);
  });

  it('nextPow2 rounds up', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(1000)).toBe(1024);
  });

  it("Parseval's identity holds (within 1%)", () => {
    const N = 512;
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = Math.random() - 0.5;
    let timeEnergy = 0;
    for (let n = 0; n < N; n++) timeEnergy += x[n] * x[n];
    const mag = fftMagnitude(x);
    let freqEnergy = 0;
    for (let k = 0; k < N; k++) freqEnergy += mag[k] * mag[k];
    freqEnergy /= N;
    expect(Math.abs(freqEnergy - timeEnergy) / timeEnergy).toBeLessThan(0.01);
  });
});
