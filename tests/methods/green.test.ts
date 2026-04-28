import { describe, it, expect } from 'vitest';
import { green } from '../../src/core/methods/green';
import { syntheticTrace } from '../helpers/synthetic';
import { estimateHr } from '../../src/core/hr/estimate';
import { filtfilt, bandpassBiquad } from '../../src/core/dsp/butterworth';
import { detrend } from '../../src/core/dsp/detrend';

// Apply the post-method pipeline (detrend + bandpass) to raw method output, then estimate.
function pipelineBpm(pulse: Float32Array, fps: number): number {
  let x = detrend(pulse, 100);
  const { b, a } = bandpassBiquad(2, 0.7, 4, fps);
  x = filtfilt(b, a, x);
  return estimateHr(x, fps).bpm;
}

describe('GREEN', () => {
  for (const fps of [15, 30, 60]) for (const bpm of [50, 70, 90, 110, 130]) {
    it(`recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec: 15, snrDb: 10 });
      const pulse = green(trace);
      const recovered = pipelineBpm(pulse, fps);
      expect(Math.abs(recovered - bpm)).toBeLessThan(2);
    });
  }
});
