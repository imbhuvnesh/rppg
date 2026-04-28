import type { RgbTrace, RppgMethod } from '../types';
import { mean, std } from '../dsp/normalize';

/**
 * GREEN method (Verkruysse et al. 2008).
 *
 * Returns z-scored, sign-flipped green channel: `pulse = -(g - mean(g)) / std(g)`.
 * Sign flip orients positive peaks toward systole, matching CHROM/POS conventions.
 */
export const green: RppgMethod = (trace: RgbTrace): Float32Array => {
  const g = trace.g;
  const m = mean(g);
  const s = std(g, m) || 1;
  const out = new Float32Array(g.length);
  for (let i = 0; i < g.length; i++) out[i] = -((g[i] - m) / s);
  return out;
};
