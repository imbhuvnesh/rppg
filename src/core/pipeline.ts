import type { RgbTrace, RppgResult, MethodName } from './types';
import { green } from './methods/green';
import { chrom } from './methods/chrom';
import { pos } from './methods/pos';
import { ica } from './methods/ica';
import { pbv } from './methods/pbv';
import { detrend } from './dsp/detrend';
import { bandpassBiquad, filtfilt } from './dsp/butterworth';
import { estimateHr } from './hr/estimate';

const METHODS: Record<MethodName, (t: RgbTrace) => Float32Array> = {
  green,
  chrom,
  pos,
  ica,
  pbv: (t) => pbv(t),
};

const FILTER_BANDS: Record<MethodName, [number, number, 1 | 2]> = {
  green: [0.7, 4.0, 1],
  // CHROM does its own 0.7-2.5 Hz internal Xs/Ys bandpass — outer band only needs
  // to cover the HR range; matching other methods at 0.7-4 Hz, order 1.
  chrom: [0.7, 4.0, 1],
  pos: [0.75, 3.0, 1],
  ica: [0.7, 4.0, 1],
  pbv: [0.7, 4.0, 1],
};

/**
 * End-to-end rPPG pipeline: method -> detrend -> bandpass -> HR estimation.
 *
 * The bandpass band/order is method-specific (FILTER_BANDS). Order 1 (2nd-order
 * final) is used everywhere except CHROM, where order 2 (4th-order final) lines
 * up with rPPG-Toolbox's CHROME_DEHAAN convention. With the RBJ-biquad-cascade
 * design, order 2 has a deeper response dip near the band edges; the per-method
 * tests confirm tolerances hold across the 5 BPM x 3 fps grid.
 */
export function pipeline(trace: RgbTrace, method: MethodName): RppgResult {
  let pulse = METHODS[method](trace);
  pulse = detrend(pulse, 100);
  const [lo, hi, order] = FILTER_BANDS[method];
  const { b, a } = bandpassBiquad(order, lo, hi, trace.fps);
  pulse = filtfilt(b, a, pulse);
  const { bpm, snr, confidence } = estimateHr(pulse, trace.fps);
  return { pulseSignal: pulse, bpm, snr, confidence };
}
