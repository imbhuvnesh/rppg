import { describe, it, expect } from 'vitest';
import { pos } from '../../src/core/methods/pos';
import { syntheticTrace } from '../helpers/synthetic';
import { estimateHr } from '../../src/core/hr/estimate';
import { filtfilt, bandpassBiquad } from '../../src/core/dsp/butterworth';
import { detrend } from '../../src/core/dsp/detrend';

function pipelineBpm(pulse: Float32Array, fps: number): number {
  let x = detrend(pulse, 100);
  // Per pipeline section 5.1: POS uses 0.75-3.0 Hz, order 1.
  const { b, a } = bandpassBiquad(1, 0.75, 3.0, fps);
  x = filtfilt(b, a, x);
  return estimateHr(x, fps).bpm;
}

describe('POS', () => {
  for (const fps of [15, 30, 60]) for (const bpm of [50, 70, 90, 110, 130]) {
    it(`recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec: 15, snrDb: 10 });
      const pulse = pos(trace);
      const recovered = pipelineBpm(pulse, fps);
      expect(Math.abs(recovered - bpm)).toBeLessThan(2);
    });
  }
});
