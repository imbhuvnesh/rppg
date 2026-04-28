import { describe, it, expect } from 'vitest';
import { chrom } from '../../src/core/methods/chrom';
import { syntheticTrace } from '../helpers/synthetic';
import { estimateHr } from '../../src/core/hr/estimate';
import { filtfilt, bandpassBiquad } from '../../src/core/dsp/butterworth';
import { detrend } from '../../src/core/dsp/detrend';

function pipelineBpm(pulse: Float32Array, fps: number): number {
  let x = detrend(pulse, 100);
  // Post-method bandpass 0.7-4 Hz, order 1 (2nd-order final). Order 2 (4th-order)
  // RBJ-biquad-cascade has a deep response dip near the lower band edge that
  // catastrophically attenuates 0.833 Hz (50 BPM) at fps=60, where CHROM's noise
  // amplification is already pushing SNR margin. Order 1 keeps the same band
  // and preserves edge frequencies. (A true bilinear-transformed Butterworth
  // would not have this dip — the RBJ biquad bandpass is a different design.)
  const { b, a } = bandpassBiquad(1, 0.7, 4, fps);
  x = filtfilt(b, a, x);
  return estimateHr(x, fps).bpm;
}

describe('CHROM', () => {
  for (const fps of [15, 30, 60]) for (const bpm of [50, 70, 90, 110, 130]) {
    it(`recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec: 15, snrDb: 10 });
      const pulse = chrom(trace);
      const recovered = pipelineBpm(pulse, fps);
      expect(Math.abs(recovered - bpm)).toBeLessThan(2);
    });
  }
});
