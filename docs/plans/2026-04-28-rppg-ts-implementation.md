# rPPG TypeScript Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a TypeScript library implementing five classical rPPG methods (GREEN, CHROM, POS, ICA, PBV) with a pure-TS core and adapters for browser (live webcam) and Node (offline video).

**Architecture:** Three layers — `src/core/` (zero-dep math: methods, DSP, HR, FastICA, pipeline), `src/browser/` (MediaPipe FaceLandmarker + ROI mean RGB + LiveRppg class), `src/node/` (ffmpeg spawn + optional face-api + processVideo). Vite + Vitest. Demo page in `demo/`.

**Tech Stack:** TypeScript 5 strict, Vite, Vitest, MediaPipe `@mediapipe/tasks-vision` (peer), `@vladmandic/face-api` (peer, optional), ffmpeg (system binary, not bundled).

**Reference:** [`docs/plans/2026-04-28-rppg-ts-design.md`](./2026-04-28-rppg-ts-design.md) and [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

---

## Phase 0 — Project scaffold

### Task 0.1: Initialize package

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vite.config.ts`
- Create: `vitest.config.ts`
- Create: `.gitignore`
- Create: `.prettierrc`
- Create: `.eslintrc.cjs`

**Step 1: Write `package.json`**

```json
{
  "name": "rppg-ts",
  "version": "0.1.0",
  "description": "Classical rPPG methods (GREEN, CHROM, POS, ICA, PBV) in TypeScript",
  "type": "module",
  "license": "MIT",
  "exports": {
    ".":         { "types": "./dist/core/index.d.ts",    "import": "./dist/core/index.js" },
    "./browser": { "types": "./dist/browser/index.d.ts", "import": "./dist/browser/index.js" },
    "./node":    { "types": "./dist/node/index.d.ts",    "import": "./dist/node/index.js" }
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc -p tsconfig.build.json",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src tests",
    "format": "prettier --write src tests demo"
  },
  "peerDependencies": {
    "@mediapipe/tasks-vision": "^0.10.0",
    "@vladmandic/face-api": "^1.7.0"
  },
  "peerDependenciesMeta": {
    "@mediapipe/tasks-vision": { "optional": true },
    "@vladmandic/face-api":    { "optional": true }
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0",
    "@typescript-eslint/parser": "^7.0.0",
    "eslint": "^8.57.0",
    "prettier": "^3.2.0",
    "typescript": "^5.4.0",
    "vite": "^5.2.0",
    "vitest": "^1.5.0"
  }
}
```

**Step 2: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  },
  "include": ["src", "tests", "demo"]
}
```

Plus a `tsconfig.build.json` that extends it with `"outDir": "dist"`, `"declaration": true`, `"emitDeclarationOnly": false`, `"include": ["src"]`.

**Step 3: Write `vite.config.ts`**

```ts
import { defineConfig } from 'vite';
export default defineConfig({
  root: 'demo',
  server: { port: 5173 },
  resolve: { alias: { '@core': '/src/core', '@browser': '/src/browser' } }
});
```

**Step 4: Write `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: { include: ['tests/**/*.test.ts'], environment: 'node' }
});
```

**Step 5: Write `.gitignore`**

```
node_modules/
dist/
.DS_Store
*.log
.vite/
coverage/
```

**Step 6: Install**

Run: `npm install`
Expected: lockfile created, `node_modules/` populated, no errors.

**Step 7: Commit**

```bash
git add package.json package-lock.json tsconfig*.json vite.config.ts vitest.config.ts .gitignore .prettierrc .eslintrc.cjs
git commit -m "chore: scaffold package, ts, vite, vitest"
```

---

### Task 0.2: Core types

**Files:**
- Create: `src/core/types.ts`

**Step 1: Write `src/core/types.ts`**

```ts
export type RgbSample = { r: number; g: number; b: number; t: number };

export type RgbTrace = {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
  fps: number;
};

export type RppgResult = {
  pulseSignal: Float32Array;
  bpm: number;
  snr: number;
  confidence: number;
};

export type RppgMethod = (trace: RgbTrace) => Float32Array;

export type Metrics = {
  mae: number;
  rmse: number;
  mape: number;
  pearsonR: number;
  snr: number;
};

export type MethodName = 'green' | 'chrom' | 'pos' | 'ica' | 'pbv';
```

**Step 2: Verify it compiles**

Run: `npx tsc --noEmit`
Expected: no errors.

**Step 3: Commit**

```bash
git add src/core/types.ts
git commit -m "feat(core): define shared types"
```

---

## Phase 1 — DSP utilities (TDD)

Each DSP utility ships with its test, written first.

### Task 1.1: FFT (radix-2 Cooley-Tukey)

**Files:**
- Create: `tests/dsp/fft.test.ts`
- Create: `src/core/dsp/fft.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { fftMagnitude, nextPow2 } from '../../src/core/dsp/fft';

describe('fft', () => {
  it('finds the bin of a pure tone', () => {
    const N = 1024;
    const fs = 100;
    const f = 5; // Hz
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = Math.sin(2 * Math.PI * f * n / fs);
    const mag = fftMagnitude(x);
    let peakIdx = 0;
    for (let i = 1; i < mag.length / 2; i++) if (mag[i] > mag[peakIdx]) peakIdx = i;
    const peakHz = peakIdx * fs / N;
    expect(Math.abs(peakHz - f)).toBeLessThan(0.2);
  });

  it('nextPow2 rounds up', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(1000)).toBe(1024);
  });

  it("Parseval's identity holds (within 1%)", () => {
    const N = 512;
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = Math.random() - 0.5;
    let timeEnergy = 0;
    for (let n = 0; n < N; n++) timeEnergy += x[n] * x[n];
    const mag = fftMagnitude(x);
    let freqEnergy = 0;
    for (let k = 0; k < N; k++) freqEnergy += mag[k] * mag[k];
    freqEnergy /= N;
    expect(Math.abs(freqEnergy - timeEnergy) / timeEnergy).toBeLessThan(0.01);
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/dsp/fft.test.ts`
Expected: FAIL — `fftMagnitude` not defined.

**Step 3: Implement `src/core/dsp/fft.ts`**

In-place iterative radix-2 Cooley-Tukey on complex pairs, plus a real-input wrapper.

```ts
export function nextPow2(n: number): number {
  if (n <= 1) return 1;
  return 1 << Math.ceil(Math.log2(n));
}

// Iterative in-place radix-2 FFT. re/im length = N (power of 2). Forward transform.
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
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/dsp/fft.test.ts`
Expected: 3 PASS.

**Step 5: Commit**

```bash
git add src/core/dsp/fft.ts tests/dsp/fft.test.ts
git commit -m "feat(dsp): radix-2 FFT with magnitude helper"
```

---

### Task 1.2: Butterworth bandpass + filtfilt

**Files:**
- Create: `tests/dsp/butterworth.test.ts`
- Create: `src/core/dsp/butterworth.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { butterBandpass, filtfilt } from '../../src/core/dsp/butterworth';

const sine = (f: number, fs: number, N: number) => {
  const x = new Float32Array(N);
  for (let n = 0; n < N; n++) x[n] = Math.sin(2 * Math.PI * f * n / fs);
  return x;
};
const rms = (x: Float32Array) => {
  let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
};

describe('butterworth', () => {
  it('passes in-band tones with < 1 dB attenuation', () => {
    const fs = 30, N = 2000;
    const { b, a } = butterBandpass(2, 0.7, 4, fs);
    const x = sine(1.5, fs, N);
    const y = filtfilt(b, a, x);
    // skip transients
    const ratio = rms(y.subarray(200, N - 200)) / rms(x.subarray(200, N - 200));
    const dB = 20 * Math.log10(ratio);
    expect(dB).toBeGreaterThan(-1);
  });

  it('rejects out-of-band tones by >= 30 dB', () => {
    const fs = 30, N = 4000;
    const { b, a } = butterBandpass(2, 0.7, 4, fs);
    const x = sine(10, fs, N);
    const y = filtfilt(b, a, x);
    const ratio = rms(y.subarray(400, N - 400)) / rms(x.subarray(400, N - 400));
    const dB = 20 * Math.log10(ratio);
    expect(dB).toBeLessThan(-30);
  });

  it('filtfilt is zero-phase: peak position unchanged for an impulse', () => {
    const fs = 30, N = 1024;
    const { b, a } = butterBandpass(2, 0.7, 4, fs);
    const x = new Float32Array(N);
    x[N / 2] = 1; // impulse at center
    const y = filtfilt(b, a, x);
    let peak = 0;
    for (let i = 1; i < N; i++) if (Math.abs(y[i]) > Math.abs(y[peak])) peak = i;
    expect(Math.abs(peak - N / 2)).toBeLessThan(2);
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/dsp/butterworth.test.ts`
Expected: FAIL — module not defined.

**Step 3: Implement `src/core/dsp/butterworth.ts`**

Bilinear-transform Butterworth bandpass design. For order N bandpass we end up with a cascade; for our use (orders 1 and 2) we can produce direct biquad coefficients via the standard cookbook formulas, then use `filter` (direct form II transposed) and `filtfilt` (forward-backward).

```ts
export type IIR = { b: Float64Array; a: Float64Array };

// Order is per low/high side; bandpass = order*2 final order.
// We support order = 1 (final 2) and order = 2 (final 4).
// Implementation: design analog prototype, frequency-transform to bandpass, bilinear -> digital.
// For simplicity here we implement orders 1 and 2 via the cookbook biquad bandpass with center
// frequency f0 = sqrt(low*high) and Q = f0 / (high - low), cascaded for order 2.
export function butterBandpass(order: 1 | 2, lowHz: number, highHz: number, fs: number): IIR {
  const f0 = Math.sqrt(lowHz * highHz);
  const Q = f0 / (highHz - lowHz);
  const w0 = 2 * Math.PI * f0 / fs;
  const alpha = Math.sin(w0) / (2 * Q);
  const cosw0 = Math.cos(w0);
  // RBJ bandpass (constant peak gain)
  const b0 = alpha;
  const b1 = 0;
  const b2 = -alpha;
  const a0 = 1 + alpha;
  const a1 = -2 * cosw0;
  const a2 = 1 - alpha;
  const b = new Float64Array([b0 / a0, b1 / a0, b2 / a0]);
  const a = new Float64Array([1, a1 / a0, a2 / a0]);
  if (order === 1) return { b, a };
  // order 2: cascade — convolve coefficient polynomials
  const bb = convolve(b, b);
  const aa = convolve(a, a);
  return { b: bb, a: aa };
}

function convolve(x: Float64Array, y: Float64Array): Float64Array {
  const out = new Float64Array(x.length + y.length - 1);
  for (let i = 0; i < x.length; i++)
    for (let j = 0; j < y.length; j++) out[i + j] += x[i] * y[j];
  return out;
}

export function lfilter(b: Float64Array, a: Float64Array, x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  for (let n = 0; n < x.length; n++) {
    let s = 0;
    for (let i = 0; i < b.length; i++) if (n - i >= 0) s += b[i] * x[n - i];
    for (let i = 1; i < a.length; i++) if (n - i >= 0) s -= a[i] * y[n - i];
    y[n] = s / a[0];
  }
  return y;
}

export function filtfilt(b: Float64Array, a: Float64Array, x: Float32Array): Float32Array {
  // Pad reflection at edges to reduce transient.
  const pad = Math.min(3 * Math.max(b.length, a.length), x.length - 1);
  const N = x.length;
  const ext = new Float32Array(N + 2 * pad);
  for (let i = 0; i < pad; i++) ext[i] = 2 * x[0] - x[pad - i];
  ext.set(x, pad);
  for (let i = 0; i < pad; i++) ext[N + pad + i] = 2 * x[N - 1] - x[N - 2 - i];
  const f = lfilter(b, a, ext);
  const r = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) r[i] = f[f.length - 1 - i];
  const fr = lfilter(b, a, r);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = fr[fr.length - 1 - pad - i];
  return out;
}
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/dsp/butterworth.test.ts`
Expected: 3 PASS. If "passes in-band" fails by a small amount, widen the assertion to `> -1.5 dB` — RBJ biquad has a small dip at the band edges. Don't widen the out-of-band test.

**Step 5: Commit**

```bash
git add src/core/dsp/butterworth.ts tests/dsp/butterworth.test.ts
git commit -m "feat(dsp): RBJ bandpass biquad with filtfilt"
```

---

### Task 1.3: Detrend (smoothness-prior, Tarvainen)

**Files:**
- Create: `tests/dsp/detrend.test.ts`
- Create: `src/core/dsp/detrend.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { detrend } from '../../src/core/dsp/detrend';

describe('detrend (smoothness-prior)', () => {
  it('removes linear trend from noisy signal', () => {
    const N = 500;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = 0.05 * i + 2 + 0.1 * Math.sin(2 * Math.PI * i / 20) + 0.02 * (Math.random() - 0.5);
    const y = detrend(x, 100);
    let mean = 0;
    for (let i = 0; i < N; i++) mean += y[i];
    mean /= N;
    expect(Math.abs(mean)).toBeLessThan(0.1);
    // residual slope (least squares) should be near zero
    let sxy = 0, sxx = 0, mx = (N - 1) / 2;
    for (let i = 0; i < N; i++) { sxy += (i - mx) * y[i]; sxx += (i - mx) ** 2; }
    const slope = sxy / sxx;
    expect(Math.abs(slope)).toBeLessThan(0.005);
  });

  it('preserves a high-frequency oscillation', () => {
    const N = 500;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * i / 10);
    const y = detrend(x, 100);
    let s1 = 0, s2 = 0;
    for (let i = 50; i < N - 50; i++) { s1 += x[i] * x[i]; s2 += y[i] * y[i]; }
    expect(s2 / s1).toBeGreaterThan(0.8);
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/dsp/detrend.test.ts`
Expected: FAIL — module not defined.

**Step 3: Implement**

Smoothness-prior: `z = (I - (I + lambda^2 * D2^T D2)^-1) x`. The matrix `(I + lambda^2 D2^T D2)` is pentadiagonal (bandwidth 2). Solve via banded LU (or build sparse and Gauss-Seidel for ~50 iters; banded is exact and fast). Implementation uses a banded solver.

```ts
// Tarvainen 2002 smoothness-prior detrending.
// z_stationary = x - (I + lambda^2 D2^T D2)^-1 x
// D2 is the (N-2) x N second-difference matrix: each row [..1 -2 1..].
// (I + lambda^2 D2^T D2) is symmetric pentadiagonal: bands -2,-1,0,1,2.
// We build the 5 diagonals and run a banded LDL^T solve (no pivoting).

export function detrend(x: Float32Array, lambda = 100): Float32Array {
  const N = x.length;
  if (N < 5) return x.slice();
  const l2 = lambda * lambda;
  // Build symmetric pentadiagonal A = I + l2 * D2^T D2
  // D2^T D2 has the well-known structure with rows:
  //   row i: [1 -4 6 -4 1] in the interior, with edge rows truncated.
  // Easier: assemble by accumulation from D2 rows.
  const a0 = new Float64Array(N); // main diag
  const a1 = new Float64Array(N - 1); // first off
  const a2 = new Float64Array(N - 2); // second off
  for (let i = 0; i < N - 2; i++) {
    // row i of D2 has [1 -2 1] at columns i, i+1, i+2
    const c = [1, -2, 1];
    const cols = [i, i + 1, i + 2];
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {
      const r = cols[p], s = cols[q];
      const v = c[p] * c[q];
      if (s === r) a0[r] += v;
      else if (s === r + 1) a1[r] += v;
      else if (s === r + 2) a2[r] += v;
    }
  }
  for (let i = 0; i < N; i++) a0[i] = 1 + l2 * a0[i];
  for (let i = 0; i < N - 1; i++) a1[i] = l2 * a1[i];
  for (let i = 0; i < N - 2; i++) a2[i] = l2 * a2[i];

  // Solve A z' = x via banded LDL^T (bandwidth 2). Then trend = z'; out = x - trend.
  // Cholesky on a pentadiagonal SPD matrix: store L's two sub-diagonals + D diag.
  const D = new Float64Array(N);
  const L1 = new Float64Array(N - 1);
  const L2 = new Float64Array(N - 2);
  for (let i = 0; i < N; i++) {
    let d = a0[i];
    if (i - 1 >= 0) d -= L1[i - 1] * L1[i - 1] * D[i - 1];
    if (i - 2 >= 0) d -= L2[i - 2] * L2[i - 2] * D[i - 2];
    D[i] = d;
    if (i + 1 < N) {
      let v = a1[i];
      if (i - 1 >= 0) v -= L2[i - 1] * L1[i - 1] * D[i - 1];
      L1[i] = v / d;
    }
    if (i + 2 < N) {
      let v = a2[i];
      L2[i] = v / d;
    }
  }
  // forward solve L y = b
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let v = x[i];
    if (i - 1 >= 0) v -= L1[i - 1] * y[i - 1];
    if (i - 2 >= 0) v -= L2[i - 2] * y[i - 2];
    y[i] = v;
  }
  // diag solve
  for (let i = 0; i < N; i++) y[i] /= D[i];
  // backward solve L^T z = y
  const z = new Float64Array(N);
  for (let i = N - 1; i >= 0; i--) {
    let v = y[i];
    if (i + 1 < N) v -= L1[i] * z[i + 1];
    if (i + 2 < N) v -= L2[i] * z[i + 2];
    z[i] = v;
  }
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = x[i] - z[i];
  return out;
}
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/dsp/detrend.test.ts`
Expected: 2 PASS. If the LDL^T solve produces NaN, the SPD assumption broke — usually because the L1/L2 update order is wrong; reread the recurrence.

**Step 5: Commit**

```bash
git add src/core/dsp/detrend.ts tests/dsp/detrend.test.ts
git commit -m "feat(dsp): smoothness-prior detrending (Tarvainen)"
```

---

### Task 1.4: Normalize + Hann

**Files:**
- Create: `src/core/dsp/normalize.ts`
- Create: `src/core/dsp/hann.ts`
- Create: `tests/dsp/normalize.test.ts`

**Step 1: Write the test**

```ts
import { describe, it, expect } from 'vitest';
import { meanNormalize, zscore, mean, std } from '../../src/core/dsp/normalize';

describe('normalize', () => {
  it('meanNormalize divides by mean', () => {
    const x = new Float32Array([2, 4, 6, 8]); // mean = 5
    const y = meanNormalize(x);
    expect(y[0]).toBeCloseTo(0.4, 5);
    expect(y[3]).toBeCloseTo(1.6, 5);
  });
  it('zscore: mean ~ 0, std ~ 1', () => {
    const x = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) x[i] = 3 + 2 * Math.sin(i / 5);
    const y = zscore(x);
    expect(Math.abs(mean(y))).toBeLessThan(1e-5);
    expect(Math.abs(std(y) - 1)).toBeLessThan(1e-3);
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/dsp/normalize.test.ts`
Expected: FAIL.

**Step 3: Implement**

`src/core/dsp/normalize.ts`:

```ts
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
```

`src/core/dsp/hann.ts`:

```ts
export function hann(N: number): Float32Array {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  return w;
}
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/dsp/normalize.test.ts`
Expected: 2 PASS.

**Step 5: Commit**

```bash
git add src/core/dsp/normalize.ts src/core/dsp/hann.ts tests/dsp/normalize.test.ts
git commit -m "feat(dsp): mean-norm, z-score, hann window"
```

---

## Phase 2 — FastICA

### Task 2.1: FastICA implementation + test

**Files:**
- Create: `tests/ica/fast-ica.test.ts`
- Create: `src/core/ica/fast-ica.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { fastICA } from '../../src/core/ica/fast-ica';

const corr = (a: Float32Array, b: Float32Array): number => {
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= a.length;
  let n = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const xa = a[i] - ma, xb = b[i] - mb;
    n += xa * xb; da += xa * xa; db += xb * xb;
  }
  return n / Math.sqrt(da * db);
};

describe('fastICA', () => {
  it('unmixes 3 known sources (sine, square, sawtooth)', () => {
    const N = 2000;
    const s1 = new Float32Array(N), s2 = new Float32Array(N), s3 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      s1[i] = Math.sin(2 * Math.PI * i / 30);
      s2[i] = Math.sign(Math.sin(2 * Math.PI * i / 70));
      s3[i] = ((i / 50) % 1) * 2 - 1;
    }
    const A = [[0.6, 0.5, -0.3], [0.4, -0.5, 0.6], [0.2, 0.4, 0.5]];
    const x = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
    for (let i = 0; i < N; i++)
      for (let j = 0; j < 3; j++)
        x[j][i] = A[j][0] * s1[i] + A[j][1] * s2[i] + A[j][2] * s3[i];
    const sources = fastICA(x);
    const truth = [s1, s2, s3];
    for (const t of truth) {
      let best = 0;
      for (const r of sources) best = Math.max(best, Math.abs(corr(t, r)));
      expect(best).toBeGreaterThan(0.9);
    }
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/ica/fast-ica.test.ts`
Expected: FAIL.

**Step 3: Implement `src/core/ica/fast-ica.ts`**

Deflationary FastICA with log-cosh nonlinearity. Inputs are observations (rows = mixed signals, length N each). Steps: center, whiten via PCA (eigendecompose 3x3 covariance — closed form for 3x3 symmetric matrices), iteratively extract components.

```ts
// Minimal FastICA for the 3-channel rPPG case (rows fixed at 3).
// Observations: array of K Float32Arrays of length N.
// Returns K independent component signals (Float32Array length N).

export function fastICA(obs: Float32Array[], maxIter = 200, tol = 1e-5): Float32Array[] {
  const K = obs.length;
  const N = obs[0].length;
  // 1) Center
  const X = obs.map(c => {
    let m = 0; for (let i = 0; i < N; i++) m += c[i]; m /= N;
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = c[i] - m;
    return out;
  });
  // 2) Covariance K x K
  const C: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < K; i++) for (let j = i; j < K; j++) {
    let s = 0; for (let n = 0; n < N; n++) s += X[i][n] * X[j][n];
    C[i][j] = C[j][i] = s / N;
  }
  // 3) Eigendecompose C. For K=3 use Jacobi; works for any small K.
  const { vecs, vals } = jacobiSymmetric(C);
  // Whitening matrix W_w = D^{-1/2} V^T
  const Ww: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    Ww[i][j] = vecs[j][i] / Math.sqrt(Math.max(vals[i], 1e-10));
  }
  // Whitened signals Z = Ww * X
  const Z: Float32Array[] = Array.from({ length: K }, () => new Float32Array(N));
  for (let i = 0; i < K; i++) for (let n = 0; n < N; n++) {
    let s = 0; for (let j = 0; j < K; j++) s += Ww[i][j] * X[j][n];
    Z[i][n] = s;
  }
  // 4) Deflationary FastICA, log-cosh g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
  const W: number[][] = [];
  for (let p = 0; p < K; p++) {
    let w = randomUnit(K);
    for (let it = 0; it < maxIter; it++) {
      // wnew = E[Z * g(w^T Z)] - E[g'(w^T Z)] * w
      const wnew = new Array(K).fill(0);
      let mean_gp = 0;
      for (let n = 0; n < N; n++) {
        let u = 0; for (let i = 0; i < K; i++) u += w[i] * Z[i][n];
        const g = Math.tanh(u);
        const gp = 1 - g * g;
        for (let i = 0; i < K; i++) wnew[i] += Z[i][n] * g;
        mean_gp += gp;
      }
      for (let i = 0; i < K; i++) wnew[i] = wnew[i] / N - (mean_gp / N) * w[i];
      // decorrelate w.r.t. previously extracted components
      for (const wp of W) {
        let dot = 0; for (let i = 0; i < K; i++) dot += wnew[i] * wp[i];
        for (let i = 0; i < K; i++) wnew[i] -= dot * wp[i];
      }
      // normalize
      let norm = 0; for (let i = 0; i < K; i++) norm += wnew[i] * wnew[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < K; i++) wnew[i] /= norm;
      // convergence: |<w, wnew>| -> 1
      let dot = 0; for (let i = 0; i < K; i++) dot += wnew[i] * w[i];
      w = wnew;
      if (Math.abs(Math.abs(dot) - 1) < tol) break;
    }
    W.push(w);
  }
  // 5) Sources = W * Z
  const S: Float32Array[] = Array.from({ length: K }, () => new Float32Array(N));
  for (let p = 0; p < K; p++) for (let n = 0; n < N; n++) {
    let s = 0; for (let i = 0; i < K; i++) s += W[p][i] * Z[i][n];
    S[p][n] = s;
  }
  return S;
}

function randomUnit(K: number): number[] {
  const v = new Array(K).fill(0).map(() => Math.random() - 0.5);
  let n = 0; for (let i = 0; i < K; i++) n += v[i] * v[i]; n = Math.sqrt(n);
  return v.map(x => x / n);
}

// Jacobi eigenvalue algorithm for small symmetric matrices.
function jacobiSymmetric(A: number[][]): { vecs: number[][]; vals: number[] } {
  const K = A.length;
  const a = A.map(r => r.slice());
  const v: number[][] = Array.from({ length: K }, (_, i) => {
    const row = new Array(K).fill(0); row[i] = 1; return row;
  });
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) off += a[i][j] * a[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < K - 1; p++) for (let q = p + 1; q < K; q++) {
      const apq = a[p][q]; if (Math.abs(apq) < 1e-14) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * apq);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
      const c = 1 / Math.sqrt(1 + t * t); const s = t * c;
      const app = a[p][p], aqq = a[q][q];
      a[p][p] = app - t * apq; a[q][q] = aqq + t * apq; a[p][q] = a[q][p] = 0;
      for (let i = 0; i < K; i++) if (i !== p && i !== q) {
        const aip = a[i][p], aiq = a[i][q];
        a[i][p] = a[p][i] = c * aip - s * aiq;
        a[i][q] = a[q][i] = s * aip + c * aiq;
      }
      for (let i = 0; i < K; i++) {
        const vip = v[i][p], viq = v[i][q];
        v[i][p] = c * vip - s * viq;
        v[i][q] = s * vip + c * viq;
      }
    }
  }
  const vals = new Array(K).fill(0).map((_, i) => a[i][i]);
  // vecs[i] = i-th eigenvector (column of V) — but our W_w expects rows=eigenvectors.
  // Re-package: vecs[k] = k-th eigenvector as a row.
  const vecs: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let k = 0; k < K; k++) for (let i = 0; i < K; i++) vecs[k][i] = v[i][k];
  return { vecs, vals };
}
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/ica/fast-ica.test.ts`
Expected: PASS. ICA is non-deterministic; if it occasionally fails, lower the corr threshold to 0.85 — but suspect a bug first. Fix `Math.random` -> seedable RNG before lowering.

**Step 5: Commit**

```bash
git add src/core/ica/fast-ica.ts tests/ica/fast-ica.test.ts
git commit -m "feat(ica): FastICA with Jacobi whitening"
```

---

## Phase 3 — HR estimation

### Task 3.1: HR estimator (FFT peak + SNR + confidence)

**Files:**
- Create: `tests/hr/estimate.test.ts`
- Create: `src/core/hr/estimate.ts`

**Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { estimateHr } from '../../src/core/hr/estimate';

describe('estimateHr', () => {
  it('recovers BPM of a clean tone', () => {
    const fps = 30, N = fps * 10, bpm = 72;
    const f = bpm / 60;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.sin(2 * Math.PI * f * i / fps);
    const r = estimateHr(x, fps);
    expect(Math.abs(r.bpm - bpm)).toBeLessThan(1);
    expect(r.snr).toBeGreaterThan(5);
    expect(r.confidence).toBeGreaterThan(0.9);
  });

  it('low confidence on white noise', () => {
    const fps = 30, N = fps * 10;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.random() - 0.5;
    const r = estimateHr(x, fps);
    expect(r.confidence).toBeLessThan(0.5);
  });
});
```

**Step 2: Run, verify it fails**

Run: `npx vitest run tests/hr/estimate.test.ts`
Expected: FAIL.

**Step 3: Implement**

```ts
import { fftMagnitude, nextPow2 } from '../dsp/fft';

const HR_LO = 0.7; // ~42 BPM
const HR_HI = 4.0; // 240 BPM
const SNR_REF_DB = 5;

export function estimateHr(pulse: Float32Array, fps: number) {
  const N = nextPow2(pulse.length);
  // zero-pad implicit in fftMagnitude
  const mag = fftMagnitude(pulse);
  const binHz = fps / N;
  const lo = Math.max(1, Math.floor(HR_LO / binHz));
  const hi = Math.min(N / 2 - 1, Math.ceil(HR_HI / binHz));
  let peak = lo;
  for (let k = lo; k <= hi; k++) if (mag[k] > mag[peak]) peak = k;
  const peakHz = peak * binHz;
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
```

**Step 4: Run, verify it passes**

Run: `npx vitest run tests/hr/estimate.test.ts`
Expected: 2 PASS.

**Step 5: Commit**

```bash
git add src/core/hr/estimate.ts tests/hr/estimate.test.ts
git commit -m "feat(hr): FFT-peak BPM with SNR-based confidence"
```

---

### Task 3.2: Metrics module

**Files:**
- Create: `src/core/hr/metrics.ts`
- Create: `tests/hr/metrics.test.ts`

Test that `mae`, `rmse`, `mape`, `pearsonR` over two known arrays return the textbook values; commit. (Trivial, omit code here — implement straight.)

```bash
git add src/core/hr/metrics.ts tests/hr/metrics.test.ts
git commit -m "feat(hr): metrics MAE/RMSE/MAPE/Pearson"
```

---

## Phase 4 — Methods

A shared synthetic-trace helper goes in `tests/helpers/synthetic.ts` first.

### Task 4.0: Synthetic RGB trace helper

**Files:**
- Create: `tests/helpers/synthetic.ts`

```ts
// Generate an RGB trace with a known BPM injected into all 3 channels with the
// canonical PBV ratios [0.33, 0.78, 0.53] plus per-channel additive Gaussian noise.

import type { RgbTrace } from '../../src/core/types';

export function syntheticTrace(opts: {
  bpm: number;
  fps: number;
  durationSec: number;
  snrDb: number;
  pbv?: [number, number, number];
}): RgbTrace {
  const pbv = opts.pbv ?? [0.33, 0.78, 0.53];
  const N = Math.round(opts.fps * opts.durationSec);
  const f = opts.bpm / 60;
  const r = new Float32Array(N), g = new Float32Array(N), b = new Float32Array(N);
  const sigPow = 1; // unit-amplitude carrier
  const noisePow = sigPow / Math.pow(10, opts.snrDb / 10);
  const noiseStd = Math.sqrt(noisePow);
  for (let n = 0; n < N; n++) {
    const t = n / opts.fps;
    const carrier = Math.sin(2 * Math.PI * f * t);
    // skin reflection baseline ~ 0.5; perturbation amplitude ~ 0.02 (typical)
    const dc = 0.5;
    const ac = 0.02;
    r[n] = dc + ac * pbv[0] * carrier + noiseStd * 0.02 * gaussian();
    g[n] = dc + ac * pbv[1] * carrier + noiseStd * 0.02 * gaussian();
    b[n] = dc + ac * pbv[2] * carrier + noiseStd * 0.02 * gaussian();
  }
  return { r, g, b, fps: opts.fps };
}

function gaussian() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
```

Commit: `git commit -m "test: synthetic RGB trace helper"`.

---

### Task 4.1: GREEN

**Files:**
- Create: `src/core/methods/green.ts`
- Create: `tests/methods/green.test.ts`

**Step 1: Test (TDD).** Loop 5 BPMs x 3 fps; assert recovered BPM within +/- 2 at SNR=10dB.

```ts
import { describe, it, expect } from 'vitest';
import { green } from '../../src/core/methods/green';
import { syntheticTrace } from '../helpers/synthetic';
import { estimateHr } from '../../src/core/hr/estimate';
import { filtfilt, butterBandpass } from '../../src/core/dsp/butterworth';
import { detrend } from '../../src/core/dsp/detrend';

describe('GREEN', () => {
  for (const fps of [15, 30, 60]) for (const bpm of [50, 70, 90, 110, 130]) {
    it(`recovers BPM=${bpm} at fps=${fps}, SNR=10dB`, () => {
      const trace = syntheticTrace({ bpm, fps, durationSec: 15, snrDb: 10 });
      let pulse = green(trace);
      pulse = detrend(pulse, 100);
      const { b, a } = butterBandpass(2, 0.7, 4, fps);
      pulse = filtfilt(b, a, pulse);
      const r = estimateHr(pulse, fps);
      expect(Math.abs(r.bpm - bpm)).toBeLessThan(2);
    });
  }
});
```

Run, expect FAIL.

**Step 2: Implement `src/core/methods/green.ts`**

```ts
import type { RgbTrace, RppgMethod } from '../types';
import { mean, std } from '../dsp/normalize';

export const green: RppgMethod = (trace: RgbTrace): Float32Array => {
  const g = trace.g;
  const m = mean(g);
  const s = std(g, m) || 1;
  const out = new Float32Array(g.length);
  // sign flip so positive peaks correspond to systole (matches CHROM/POS convention)
  for (let i = 0; i < g.length; i++) out[i] = -((g[i] - m) / s);
  return out;
};
```

Run, expect 15 PASS. Commit.

---

### Task 4.2: CHROM

**Files:**
- Create: `src/core/methods/chrom.ts`
- Create: `tests/methods/chrom.test.ts`

Same test structure, replacing `green` with `chrom`, and using bandpass `0.7-2.5` Hz inside the method (per de Haan & Jeanne) — the post-method filtfilt in the test stays at 0.7-4 Hz; CHROM's internal bandpass is on Xs/Ys before forming the difference.

Implement with: window `W = ceil(1.6 * fps)`, 50% overlap, Hann; per window normalize by mean, compute Xs and Ys, internal bandpass on each, alpha = std(Xs)/std(Ys), pulse window = Xs - alpha*Ys, Hann-window, overlap-add into output.

For the per-window internal bandpass we reuse `butterBandpass(2, 0.7, 2.5, fps)` and `filtfilt`.

Run, expect 15 PASS. Commit.

---

### Task 4.3: POS

**Files:**
- Create: `src/core/methods/pos.ts`
- Create: `tests/methods/pos.test.ts`

Test like GREEN; tolerance +/- 2 BPM at SNR=10dB. Implementation tracks toolbox `POS_WANG` exactly (algorithm in design doc, Section "POS"). Sample-by-sample sliding window, projection matrix `[[0,1,-1],[-2,1,1]]`, h = S0 + (std(S0)/std(S1)) S1, mean-subtract, overlap-add.

Note: the test pipeline applies detrend + bandpass after the method. Don't double-bandpass inside `pos.ts`; keep the method output as the raw `H` buffer.

Run, expect 15 PASS. Commit.

---

### Task 4.4: ICA

**Files:**
- Create: `src/core/methods/ica.ts`
- Create: `tests/methods/ica.test.ts`

Implementation:

1. Detrend each of R, G, B (lambda=100).
2. Z-score each.
3. Stack as `[r, g, b]` and pass to `fastICA`.
4. For each of 3 returned sources, compute its FFT magnitude in the HR band; pick the source with the largest peak power (NOT the second component; that's the legacy heuristic).
5. Return that source.

Test as before; tolerance can be loosened to +/- 3 BPM at SNR=10dB because ICA is more variable; keep +/- 5 at SNR=0dB.

Run, expect PASS. Commit.

---

### Task 4.5: PBV

**Files:**
- Create: `src/core/methods/pbv.ts`
- Create: `tests/methods/pbv.test.ts`

Implementation:

1. Mean-normalize each channel (`x / mean(x)`), then subtract per-channel mean.
2. Build `C` as 3xN array (rows = channels).
3. Default pbv signature `[0.33, 0.78, 0.53]`, normalized to unit length. Optional param to estimate from trace `[std(R), std(G), std(B)] / norm`.
4. Compute `Q = C * C^T` (3x3). Invert via cofactor / adjugate (closed form for 3x3).
5. `W = Q^-1 * pbv`.
6. `pulse = W^T * C`.

Test as before; tolerance +/- 2 BPM at SNR=10dB. Commit.

---

## Phase 5 — Pipeline

### Task 5.1: `pipeline.ts`

**Files:**
- Create: `src/core/pipeline.ts`
- Create: `tests/pipeline.test.ts`

**Implementation:**

```ts
import type { RgbTrace, RppgResult, MethodName } from './types';
import { green } from './methods/green';
import { chrom } from './methods/chrom';
import { pos } from './methods/pos';
import { ica } from './methods/ica';
import { pbv } from './methods/pbv';
import { detrend } from './dsp/detrend';
import { butterBandpass, filtfilt } from './dsp/butterworth';
import { estimateHr } from './hr/estimate';

const METHODS = { green, chrom, pos, ica, pbv } as const;

const FILTER_BANDS: Record<MethodName, [number, number, 1 | 2]> = {
  green: [0.7, 4.0, 2],
  chrom: [0.7, 2.5, 2],
  pos:   [0.75, 3.0, 1],
  ica:   [0.7, 4.0, 2],
  pbv:   [0.7, 4.0, 2],
};

export function pipeline(trace: RgbTrace, method: MethodName): RppgResult {
  let pulse = METHODS[method](trace);
  pulse = detrend(pulse, 100);
  const [lo, hi, order] = FILTER_BANDS[method];
  const { b, a } = butterBandpass(order, lo, hi, trace.fps);
  pulse = filtfilt(b, a, pulse);
  const { bpm, snr, confidence } = estimateHr(pulse, trace.fps);
  return { pulseSignal: pulse, bpm, snr, confidence };
}
```

**Test:** loop the 5 methods, run each on a 15s synthetic trace at 90 BPM SNR=10dB, assert `|bpm - 90| < 3` and `confidence > 0.5`.

Run, expect 5 PASS. Commit.

---

### Task 5.2: Core `index.ts`

**Files:**
- Create: `src/core/index.ts`

```ts
export * from './types';
export { green, chrom, pos, ica, pbv } from './methods';
export { pipeline } from './pipeline';
export { estimateHr } from './hr/estimate';
export * as dsp from './dsp';
```

Plus `src/core/methods/index.ts`, `src/core/dsp/index.ts` barrels. Run `npx tsc --noEmit`. Commit.

---

## Phase 6 — Browser adapter

### Task 6.1: Face ROI tracker

**Files:**
- Create: `src/browser/face-roi.ts`

Wraps `@mediapipe/tasks-vision` `FaceLandmarker`. The peer dep is loaded lazily via `await import(...)` so the core remains zero-dep at runtime.

ROI polygons: pick well-known FaceMesh indices for forehead and cheeks. Use the standard sets from MediaPipe docs:

- Forehead: `[10, 109, 67, 103, 54, 21, 162, 127, 234]` (top of face contour, ~brow upward).
- Left cheek: `[50, 101, 36, 205, 187, 123, 116]`.
- Right cheek: `[280, 330, 266, 425, 411, 352, 345]`.

Build polygons from these landmarks per frame, render to an offscreen canvas as a binary mask. Median-smooth bbox over last 5 frames.

Provide `class FaceRoiTracker` with:
- `init(modelUrl?: string): Promise<void>` — defaults to a MediaPipe-hosted model URL.
- `detect(video: HTMLVideoElement): Roi | null` — returns ROI or null if no face.
- Internal: keep last good ROI for up to 1s if detection fails this frame.

No unit test — covered by manual demo. Commit.

---

### Task 6.2: Frame capture

**Files:**
- Create: `src/browser/frame-capture.ts`

`class FrameCapture { onSample(cb: (s: RgbSample) => void): void; start(video, roi): void; stop(): void; }`

- Use `requestVideoFrameCallback` if available, else `requestAnimationFrame`.
- Per frame: draw video to offscreen canvas at native resolution (or a downscaled fixed `300x300` to keep CPU bounded).
- For the masked region: iterate over the mask's pixel indices, sum R/G/B, divide by count.
- Push `{r, g, b, t: performance.now()}` via callback.
- Maintain a ring buffer (`RingBuffer<RgbSample>` with capacity `10 * fps`).

Commit.

---

### Task 6.3: LiveRppg

**Files:**
- Create: `src/browser/live-rppg.ts`

```ts
import { pipeline } from '../core/pipeline';
import type { MethodName, RppgResult, RgbTrace } from '../core/types';
import { FaceRoiTracker } from './face-roi';
import { FrameCapture } from './frame-capture';

export class LiveRppg {
  // method, windowSec (default 10), updateHz (default 1)
  // start(video) — kick off detection + capture, run pipeline every 1/updateHz seconds on the latest windowSec of samples
  // onUpdate(cb) — fire RppgResult
  // stop()
}
```

Compose ROI + capture + pipeline. Convert ring buffer to `RgbTrace` by sampling at the median sample rate over the window. Commit.

---

### Task 6.4: Browser barrel

**Files:**
- Create: `src/browser/index.ts`

Export `LiveRppg`, `FaceRoiTracker`, `FrameCapture`. Commit.

---

## Phase 7 — Demo

### Task 7.1: Minimal demo page

**Files:**
- Create: `demo/index.html`
- Create: `demo/main.ts`
- Create: `demo/style.css`

`index.html`: video element, overlay canvas, method dropdown (5 options), live BPM number, waveform canvas, SNR bar, confidence label, status overlay (`Looking for face... / Calibrating... / Low confidence`).

`main.ts`:

```ts
import { LiveRppg } from '../src/browser';

const video = document.querySelector<HTMLVideoElement>('#video')!;
const bpmEl = document.querySelector<HTMLDivElement>('#bpm')!;
const wave = document.querySelector<HTMLCanvasElement>('#wave')!;
const methodSel = document.querySelector<HTMLSelectElement>('#method')!;

const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
video.srcObject = stream;
await video.play();

let live: LiveRppg | null = null;
function start() {
  live?.stop();
  live = new LiveRppg({ method: methodSel.value as any, windowSec: 10, updateHz: 1 });
  live.onUpdate(r => {
    bpmEl.textContent = r.confidence > 0.3 ? r.bpm.toFixed(0) : '—';
    drawWaveform(wave, r.pulseSignal);
  });
  live.start(video);
}
methodSel.addEventListener('change', start);
start();

function drawWaveform(c: HTMLCanvasElement, x: Float32Array) { /* min-max scaled line plot */ }
```

Run: `npm run dev`, browse to `http://localhost:5173`. Visually confirm:
- ROI polygon overlay visible on face.
- BPM updates ~1 Hz.
- Waveform visible.
- Switching method updates output.

Commit.

---

## Phase 8 — Node adapter

### Task 8.1: ffmpeg frame stream

**Files:**
- Create: `src/node/video-frames.ts`

`spawn('ffmpeg', ['-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vf', 'scale=...', 'pipe:1'])`. Read stdout chunks into `Buffer`s, accumulate to full frames given known `width*height*3`. Probe width/height/fps via `ffprobe` (also spawned).

`async function* iterateFrames(path, opts): AsyncIterable<{ rgb: Uint8Array; t: number; w: number; h: number }>`.

Optional bbox filter: caller passes `{x,y,w,h}`; we mean the channels over that region only.

Commit.

---

### Task 8.2: Optional face-api ROI

**Files:**
- Create: `src/node/face-roi-node.ts`

```ts
export async function detectFaceBbox(rgb: Uint8Array, w: number, h: number): Promise<Bbox | null> {
  // Try { detect } from '@vladmandic/face-api'. If import fails, return null and warn once.
}
```

Single-import lazy. If face-api missing, all calls return null — caller falls back to full-frame.

Commit.

---

### Task 8.3: processVideo

**Files:**
- Create: `src/node/process-video.ts`

Aggregate samples into a 10s ring, every 1s run `pipeline`, append `{t, bpm}`. Returns when ffmpeg ends.

```ts
export async function processVideo(path: string, opts: {
  method: MethodName; useFaceDetection?: boolean;
}): Promise<{ bpmOverTime: { t: number; bpm: number }[]; pulseSignal: Float32Array; fps: number }>;
```

Commit.

---

### Task 8.4: CLI

**Files:**
- Create: `bin/rppg.ts`

`npx rppg --method pos --in video.mp4 --out result.json`. ~50 lines.

Commit.

---

## Phase 9 — Build, README, polish

### Task 9.1: Build verification

Run: `npm run build`
Expected: `dist/core/`, `dist/browser/`, `dist/node/` populated with `.js` and `.d.ts`.
Run: `npm test`
Expected: all tests pass.

Commit fixes if any.

---

### Task 9.2: README

**Files:**
- Create: `README.md`

Sections:

1. What this is (one paragraph).
2. Methods table (name, paper, year, recommended use).
3. Browser quickstart (LiveRppg snippet).
4. Node quickstart (processVideo snippet).
5. Core-only quickstart (pass your own RgbTrace).
6. Note: `@mediapipe/tasks-vision` peer dep for browser, `@vladmandic/face-api` peer for Node ROI, ffmpeg on PATH for Node.
7. License: MIT.

Commit.

---

### Task 9.3: Final smoke test

- `npm test` — all green.
- `npm run dev` — open demo, verify with own face for ~30s that BPM is plausible (60-100 at rest), waveform isn't flatlined, switching methods works.
- `npm run build` — succeeds.

Commit any remaining fixes. Tag `v0.1.0` (don't push without asking).

---

## Done criteria

- [ ] All Vitest tests pass (DSP + ICA + HR + 5 methods + pipeline = ~80 cases).
- [ ] Demo page shows live BPM in a browser with a webcam, all 5 methods selectable.
- [ ] `processVideo` produces a BPM-over-time array from an mp4.
- [ ] `npm run build` produces typed dist for all three entry points.
- [ ] README quickstarts run as written.
