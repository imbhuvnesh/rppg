import { fftMagnitude, nextPow2 } from '../dsp/fft';

const HR_LO = 0.7; // ~42 BPM
const HR_HI = 4.0; // 240 BPM
const SNR_REF_DB = 5;

/**
 * Estimate heart rate from a 1-D pulse signal via FFT-peak detection.
 *
 * Assumes `pulse` is already detrended and bandpassed to a heart-rate-relevant band.
 * Internally zero-pads to nextPow2; sub-bin resolution achieved via 3-point parabolic
 * interpolation around the peak.
 *
 * Returns:
 *   - `bpm`: heart rate in beats per minute (peak frequency × 60).
 *   - `snr`: signal-to-noise ratio in dB. Numerator = power within ±0.1 Hz of the peak
 *           and its first harmonic. Denominator = remaining in-band power [0.7, 4.0] Hz.
 *   - `confidence`: clamp(snr / 5 dB, 0, 1).
 */
export function estimateHr(pulse: Float32Array, fps: number) {
  const N = nextPow2(pulse.length);
  // zero-pad implicit in fftMagnitude
  const mag = fftMagnitude(pulse);
  const binHz = fps / N;
  const lo = Math.max(1, Math.floor(HR_LO / binHz));
  const hi = Math.min(N / 2 - 1, Math.ceil(HR_HI / binHz));
  let peak = lo;
  for (let k = lo; k <= hi; k++) if (mag[k] > mag[peak]) peak = k;
  // Parabolic (quadratic) interpolation around the peak for sub-bin frequency
  // resolution. With 10 s @ 30 fps padded to 512, raw bin spacing is ~3.5 BPM,
  // which is too coarse for the <1 BPM tolerance demanded by the tests.
  let peakHz = peak * binHz;
  if (peak > lo && peak < hi) {
    const yL = mag[peak - 1], yC = mag[peak], yR = mag[peak + 1];
    const denom = yL - 2 * yC + yR;
    if (denom !== 0) {
      const delta = 0.5 * (yL - yR) / denom;
      peakHz = (peak + delta) * binHz;
    }
  }
  const bpm = peakHz * 60;
  // SNR: energy in +/- 0.1 Hz of peak + first harmonic vs. rest of band
  const tol = Math.max(1, Math.round(0.1 / binHz));
  const harm = Math.round(2 * peak);
  let sig = 0, noise = 0;
  for (let k = lo; k <= hi; k++) {
    const power = mag[k] * mag[k];
    const inPeak = Math.abs(k - peak) <= tol;
    const inHarm = harm <= hi && Math.abs(k - harm) <= tol;
    if (inPeak || inHarm) sig += power; else noise += power;
  }
  const snrLin = sig / Math.max(noise, 1e-12);
  const snr = 10 * Math.log10(snrLin);
  const confidence = Math.max(0, Math.min(1, snr / SNR_REF_DB));
  return { bpm, snr, confidence };
}
