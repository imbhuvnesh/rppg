import { describe, it, expect } from 'vitest';
import { meanNormalize, zscore, mean, std } from '../../src/core/dsp/normalize';

describe('normalize', () => {
  it('meanNormalize divides by mean', () => {
    const x = new Float32Array([2, 4, 6, 8]); // mean = 5
    const y = meanNormalize(x);
    expect(y[0]).toBeCloseTo(0.4, 5);
    expect(y[3]).toBeCloseTo(1.6, 5);
  });
  it('zscore: mean ~ 0, std ~ 1', () => {
    const x = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) x[i] = 3 + 2 * Math.sin(i / 5);
    const y = zscore(x);
    expect(Math.abs(mean(y))).toBeLessThan(1e-5);
    expect(Math.abs(std(y) - 1)).toBeLessThan(1e-3);
  });
});
