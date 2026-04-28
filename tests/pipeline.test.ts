import { describe, it, expect } from 'vitest';
import type { MethodName } from '../src/core/types';
import { pipeline } from '../src/core/pipeline';
import { syntheticTrace } from './helpers/synthetic';

describe('pipeline', () => {
  const methods: MethodName[] = ['green', 'chrom', 'pos', 'ica', 'pbv'];
  const bpms = [50, 70, 90, 110, 130];
  const fpsList = [15, 30, 60];
  const durationSec = 15;

  for (const m of methods) {
    for (const fps of fpsList) {
      for (const bpm of bpms) {
        it(`${m}: recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
          const trace = syntheticTrace({ bpm, fps, durationSec, snrDb: 10 });
          const r = pipeline(trace, m);
          // ICA gets a slightly looser tolerance, matching its per-method tests.
          const tol = m === 'ica' ? 4 : 3;
          expect(Math.abs(r.bpm - bpm)).toBeLessThan(tol);
          expect(r.confidence).toBeGreaterThan(0.3);
        });
      }
    }
  }
});
