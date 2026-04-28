import { describe, it, expect } from 'vitest';
import { hann } from '../../src/core/dsp/hann';
import { bandpassBiquad, filtfilt } from '../../src/core/dsp/butterworth';

describe('edge cases', () => {
  it('hann(0) returns a length-0 array', () => {
    const w = hann(0);
    expect(w.length).toBe(0);
  });

  it('hann(1) returns [1]', () => {
    const w = hann(1);
    expect(w.length).toBe(1);
    expect(w[0]).toBe(1);
  });

  it('filtfilt on empty input returns a length-0 array without throwing', () => {
    const { b, a } = bandpassBiquad(2, 0.7, 4, 30);
    const y = filtfilt(b, a, new Float32Array(0));
    expect(y.length).toBe(0);
  });
});
