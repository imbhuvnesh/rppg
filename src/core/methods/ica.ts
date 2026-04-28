import type { RgbTrace, RppgMethod } from '../types';
import { detrend } from '../dsp/detrend';
import { zscore } from '../dsp/normalize';
import { fastICA } from '../ica/fast-ica';
import { fftMagnitude, nextPow2 } from '../dsp/fft';

const HR_LO = 0.7; // ~42 BPM
const HR_HI = 4.0; // 240 BPM

/**
 * ICA method (Poh, McDuff, Picard 2010 / 2011).
 *
 * Detrend each of R, G, B (lambda=100), z-score per channel, stack as 3xN, run
 * FastICA. Pick the source with the largest FFT-magnitude peak in [0.7, 4.0] Hz
 * (NOT the second component — that's the legacy heuristic). Return that source.
 */
export const ica: RppgMethod = (trace: RgbTrace): Float32Array => {
  const { r, g, b, fps } = trace;

  // Detrend then z-score each channel.
  const rPre = zscore(detrend(r, 100));
  const gPre = zscore(detrend(g, 100));
  const bPre = zscore(detrend(b, 100));

  const sources = fastICA([rPre, gPre, bPre]);

  // Pick the source with the largest FFT-magnitude peak in [HR_LO, HR_HI] Hz.
  let bestIdx = 0;
  let bestPeak = -1;
  for (let s = 0; s < sources.length; s++) {
    const sig = sources[s];
    const N = nextPow2(sig.length);
    const mag = fftMagnitude(sig);
    const binHz = fps / N;
    const lo = Math.max(1, Math.floor(HR_LO / binHz));
    const hi = Math.min(N / 2 - 1, Math.ceil(HR_HI / binHz));
    let peak = 0;
    for (let k = lo; k <= hi; k++) if (mag[k] > peak) peak = mag[k];
    if (peak > bestPeak) {
      bestPeak = peak;
      bestIdx = s;
    }
  }
  return sources[bestIdx];
};
