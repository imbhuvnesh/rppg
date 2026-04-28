export function nextPow2(n: number): number {
  if (n <= 1) return 1;
  return 1 << Math.ceil(Math.log2(n));
}

/**
 * In-place radix-2 Cooley-Tukey forward FFT. `re` and `im` must be the same
 * power-of-2 length. Twiddles are accumulated in float64 internally even
 * though storage is float32, to limit drift over many stages.
 */
export function fftInPlace(re: Float32Array, im: Float32Array): void {
  const N = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < half; k++) {
        const tRe = curRe * re[i + k + half] - curIm * im[i + k + half];
        const tIm = curRe * im[i + k + half] + curIm * re[i + k + half];
        re[i + k + half] = re[i + k] - tRe;
        im[i + k + half] = im[i + k] - tIm;
        re[i + k] += tRe;
        im[i + k] += tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/**
 * Magnitude spectrum of a real signal. Zero-pads to `nextPow2(x.length)`,
 * so `mag.length === nextPow2(x.length)` and bin k corresponds to frequency
 * `k * fs / mag.length` (NOT `k * fs / x.length`). Returns full N bins; use
 * the first N/2 for unique frequencies.
 */
export function fftMagnitude(x: Float32Array): Float32Array {
  const N = nextPow2(x.length);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(x);
  fftInPlace(re, im);
  const mag = new Float32Array(N);
  for (let k = 0; k < N; k++) mag[k] = Math.hypot(re[k], im[k]);
  return mag;
}
