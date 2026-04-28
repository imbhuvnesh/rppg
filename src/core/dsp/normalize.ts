/** Mean of a Float32Array. */
export function mean(x: Float32Array): number {
  let s = 0; for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}
/** Population standard deviation (denominator N). Pass `m` if you've already computed the mean. */
export function std(x: Float32Array, m = mean(x)): number {
  let s = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - m; s += d * d; }
  return Math.sqrt(s / x.length);
}
/** Divide by the mean. Returns NaN-filled output if mean is 0. */
export function meanNormalize(x: Float32Array): Float32Array {
  const m = mean(x);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / m;
  return out;
}
/** Z-score: (x - mean) / std. Guarded against zero std (treats it as 1). */
export function zscore(x: Float32Array): Float32Array {
  const m = mean(x);
  const s = std(x, m) || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) / s;
  return out;
}
/** Subtract the mean. */
export function subtractMean(x: Float32Array): Float32Array {
  const m = mean(x);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - m;
  return out;
}
