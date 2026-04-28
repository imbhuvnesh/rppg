export function mean(x: Float32Array): number {
  let s = 0; for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}
export function std(x: Float32Array, m = mean(x)): number {
  let s = 0; for (let i = 0; i < x.length; i++) { const d = x[i] - m; s += d * d; }
  return Math.sqrt(s / x.length);
}
export function meanNormalize(x: Float32Array): Float32Array {
  const m = mean(x);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / m;
  return out;
}
export function zscore(x: Float32Array): Float32Array {
  const m = mean(x);
  const s = std(x, m) || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) / s;
  return out;
}
export function subtractMean(x: Float32Array): Float32Array {
  const m = mean(x);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - m;
  return out;
}
