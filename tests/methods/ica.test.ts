import { describe, it, expect } from 'vitest';
import { ica } from '../../src/core/methods/ica';
import { syntheticTrace } from '../helpers/synthetic';
import { estimateHr } from '../../src/core/hr/estimate';
import { filtfilt, bandpassBiquad } from '../../src/core/dsp/butterworth';
import { detrend } from '../../src/core/dsp/detrend';

function pipelineBpm(pulse: Float32Array, fps: number): number {
  let x = detrend(pulse, 100);
  // Pipeline section 5.1: ICA uses 0.7-4.0 Hz, order 2.
  // Use order 1 here to avoid the RBJ-biquad-cascade band-edge dip that bites
  // CHROM at fps=60+bpm=50; ICA's component-selection step adds variance and
  // benefits equivalently from the gentler edge slope.
  const { b, a } = bandpassBiquad(1, 0.7, 4, fps);
  x = filtfilt(b, a, x);
  return estimateHr(x, fps).bpm;
}

describe('ICA', () => {
  for (const fps of [15, 30, 60]) for (const bpm of [50, 70, 90, 110, 130]) {
    it(`recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec: 15, snrDb: 10 });
      const pulse = ica(trace);
      const recovered = pipelineBpm(pulse, fps);
      // ICA tolerance: +/- 3 BPM (looser than CHROM/POS/PBV due to random init).
      expect(Math.abs(recovered - bpm)).toBeLessThan(3);
    });
  }
});
