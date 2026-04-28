/** Hann window of length N. For N=0 returns an empty array; for N=1 returns [1]. */
export function hann(N: number): Float32Array {
  if (N < 2) return new Float32Array(N === 1 ? [1] : []);
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  return w;
}
