import { describe, it, expect } from 'vitest';
import type { MethodName } from '../src/core/types';
import { pipeline } from '../src/core/pipeline';
import { syntheticTrace } from './helpers/synthetic';

describe('pipeline', () => {
  const methods: MethodName[] = ['green', 'chrom', 'pos', 'ica', 'pbv'];
  const fps = 30, durationSec = 15, bpm = 90;

  for (const m of methods) {
    it(`${m}: recovers BPM=${bpm} at SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec, snrDb: 10 });
      const r = pipeline(trace, m);
      expect(Math.abs(r.bpm - bpm)).toBeLessThan(3);
      expect(r.confidence).toBeGreaterThan(0.3);
    });
  }
});
