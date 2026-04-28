/** Mean absolute error. */
export function mae(predicted: number[], actual: number[]): number {
  if (predicted.length !== actual.length) throw new Error('length mismatch');
  const n = predicted.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.abs(predicted[i] - actual[i]);
  return s / n;
}

/** Root mean squared error. */
export function rmse(predicted: number[], actual: number[]): number {
  if (predicted.length !== actual.length) throw new Error('length mismatch');
  const n = predicted.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = predicted[i] - actual[i];
    s += d * d;
  }
  return Math.sqrt(s / n);
}

/** Mean absolute percentage error (in %; e.g. 5.0 for a 5% error). Skips entries where actual === 0. */
export function mape(predicted: number[], actual: number[]): number {
  if (predicted.length !== actual.length) throw new Error('length mismatch');
  const n = predicted.length;
  let s = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (actual[i] === 0) continue;
    s += Math.abs((predicted[i] - actual[i]) / actual[i]);
    count++;
  }
  if (count === 0) return 0;
  return (s / count) * 100;
}

/** Pearson correlation coefficient. Returns 0 if either sample has zero variance. */
export function pearsonR(predicted: number[], actual: number[]): number {
  if (predicted.length !== actual.length) throw new Error('length mismatch');
  const n = predicted.length;
  if (n === 0) return 0;
  let mp = 0, ma = 0;
  for (let i = 0; i < n; i++) { mp += predicted[i]; ma += actual[i]; }
  mp /= n; ma /= n;
  let num = 0, dp = 0, da = 0;
  for (let i = 0; i < n; i++) {
    const xp = predicted[i] - mp;
    const xa = actual[i] - ma;
    num += xp * xa;
    dp += xp * xp;
    da += xa * xa;
  }
  if (dp === 0 || da === 0) return 0;
  return num / Math.sqrt(dp * da);
}
