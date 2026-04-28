import { describe, it, expect } from 'vitest';
import { mae, rmse, mape, pearsonR } from '../../src/core/hr/metrics';

describe('metrics', () => {
  it('returns zero error and unit correlation for identical arrays', () => {
    const a = [60, 70, 80, 90, 100];
    expect(mae(a, a)).toBe(0);
    expect(rmse(a, a)).toBe(0);
    expect(mape(a, a)).toBe(0);
    expect(pearsonR(a, a)).toBeCloseTo(1, 10);
  });

  it('matches hand-computed values on a small known case', () => {
    const predicted = [70, 80, 90];
    const actual = [72, 78, 91];
    // |errors| = [2, 2, 1] -> MAE = 5/3
    // squared = [4, 4, 1] -> RMSE = sqrt(9/3) = sqrt(3)
    // pct errors (in %) = [2/72, 2/78, 1/91] * 100 -> mean
    const expectedMape = (2 / 72 + 2 / 78 + 1 / 91) * 100 / 3;
    expect(mae(predicted, actual)).toBeCloseTo(5 / 3, 5);
    expect(rmse(predicted, actual)).toBeCloseTo(Math.sqrt(3), 5);
    expect(mape(predicted, actual)).toBeCloseTo(expectedMape, 5);
    // Pearson should be high but not 1 here (hand-computed ~ 0.978).
    expect(pearsonR(predicted, actual)).toBeGreaterThan(0.97);
    expect(pearsonR(predicted, actual)).toBeLessThan(1);
  });

  it('pearsonR handles anti-correlation and zero variance', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [5, 4, 3, 2, 1];
    expect(pearsonR(a, b)).toBeCloseTo(-1, 10);

    const c = [7, 7, 7, 7];
    const d = [1, 2, 3, 4];
    expect(pearsonR(c, d)).toBe(0);
    expect(pearsonR(d, c)).toBe(0);
  });

  it('throws on length mismatch', () => {
    expect(() => mae([1, 2], [1, 2, 3])).toThrow('length mismatch');
    expect(() => rmse([1, 2], [1, 2, 3])).toThrow('length mismatch');
    expect(() => mape([1, 2], [1, 2, 3])).toThrow('length mismatch');
    expect(() => pearsonR([1, 2], [1, 2, 3])).toThrow('length mismatch');
  });
});
