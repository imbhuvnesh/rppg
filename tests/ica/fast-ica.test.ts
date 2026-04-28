import { describe, it, expect } from 'vitest';
import { fastICA } from '../../src/core/ica/fast-ica';

const corr = (a: Float32Array, b: Float32Array): number => {
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= a.length;
  let n = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const xa = a[i] - ma, xb = b[i] - mb;
    n += xa * xb; da += xa * xa; db += xb * xb;
  }
  return n / Math.sqrt(da * db);
};

describe('fastICA', () => {
  it('unmixes 3 known sources (sine, square, sawtooth)', () => {
    const N = 2000;
    const s1 = new Float32Array(N), s2 = new Float32Array(N), s3 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      s1[i] = Math.sin(2 * Math.PI * i / 30);
      s2[i] = Math.sign(Math.sin(2 * Math.PI * i / 70));
      s3[i] = ((i / 50) % 1) * 2 - 1;
    }
    const A = [[0.6, 0.5, -0.3], [0.4, -0.5, 0.6], [0.2, 0.4, 0.5]];
    const x = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
    for (let i = 0; i < N; i++)
      for (let j = 0; j < 3; j++)
        x[j][i] = A[j][0] * s1[i] + A[j][1] * s2[i] + A[j][2] * s3[i];
    const sources = fastICA(x);
    const truth = [s1, s2, s3];
    for (const t of truth) {
      let best = 0;
      for (const r of sources) best = Math.max(best, Math.abs(corr(t, r)));
      expect(best).toBeGreaterThan(0.9);
    }
  });
});
