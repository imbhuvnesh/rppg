# rPPG in TypeScript — Design

Date: 2026-04-28
Status: Approved (brainstorm complete)
Reference: [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)

## Goal

A TypeScript library that implements the five canonical classical (signal-processing,
non-deep-learning) rPPG methods, with a pure-TS core and thin adapters for the
browser (live webcam) and Node (offline video files).

## Scope

In scope:

- Five methods: GREEN, CHROM, POS, ICA, PBV.
- Full pipeline: ROI mean RGB trace -> method -> detrend + bandpass -> HR/SNR/confidence.
- Browser adapter with MediaPipe FaceLandmarker, demo page (Vite).
- Node adapter with ffmpeg (spawned, not bundled), optional face-api peer dep.
- Vitest unit tests for DSP utilities and method correctness on synthetic signals.

Out of scope (v1):

- Deep-learning methods (DeepPhys, MTTS-CAN, PhysNet, etc.).
- LGI, OMIT, PCA methods.
- Public-dataset benchmarks (UBFC-rPPG, PURE).
- Web Workers / OffscreenCanvas optimization.
- Playwright / browser integration tests.

## Architecture

Three layers, each independently usable.

### Core (`src/core/`, zero runtime deps)

```
core/
  methods/   green.ts chrom.ts pos.ts ica.ts pbv.ts
  dsp/       butterworth.ts detrend.ts fft.ts normalize.ts hann.ts
  hr/        estimate.ts metrics.ts
  ica/       fast-ica.ts
  pipeline.ts
  types.ts
  index.ts
```

### Browser adapter (`src/browser/`, peer dep: `@mediapipe/tasks-vision`)

```
browser/
  face-roi.ts        FaceRoiTracker, ROI = forehead + cheeks polygons
  frame-capture.ts   requestVideoFrameCallback, ROI-masked mean RGB
  live-rppg.ts       LiveRppg class, composes the above with core/pipeline
  index.ts
```

### Node adapter (`src/node/`, peer dep: `@vladmandic/face-api` optional)

```
node/
  video-frames.ts    spawns ffmpeg (rawvideo, rgb24)
  face-roi-node.ts   optional face-api.js wrapper, full-frame fallback
  process-video.ts   processVideo(path, opts) -> { bpmOverTime, pulseSignal, fps }
  index.ts
```

### Demo (`demo/`)

```
demo/
  index.html         video, canvas overlay, method dropdown, BPM, waveform, SNR bar
  main.ts            getUserMedia -> <video> -> LiveRppg, plot on 2D canvas
```

### Types (the contract that holds it together)

```ts
type RgbSample = { r: number; g: number; b: number; t: number }; // t = ms
type RgbTrace  = { r: Float32Array; g: Float32Array; b: Float32Array; fps: number };
type RppgResult = { pulseSignal: Float32Array; bpm: number; snr: number; confidence: number };
type RppgMethod = (trace: RgbTrace) => Float32Array; // raw pulse, before post-processing
type Metrics = { mae: number; rmse: number; mape: number; pearsonR: number; snr: number };
```

## Algorithm specifications

Common preprocessing, applied by `pipeline.ts` after the method runs:

1. Detrend with smoothness-prior (Tarvainen 2002), lambda = 100.
2. Butterworth bandpass, filtfilt (zero-phase), band per method (below).
3. HR estimation: FFT, peak in [0.7, 4.0] Hz, BPM = peak * 60.
4. SNR: (energy in +/- 0.1 Hz of peak + first harmonic) / (in-band energy excluding those).
5. Confidence: clamp(SNR / SNR_ref, 0, 1) with SNR_ref = 5 dB.

### GREEN (Verkruysse 2008)

- Pulse = -(g - mean(g)) / std(g). Sign flipped so peaks correspond to systole.
- Bandpass 0.7-4 Hz, 2nd-order.

### CHROM (de Haan & Jeanne 2013)

- Sliding window W = 1.6 * fps, 50% overlap, Hann-windowed.
- Per window:
  - RGBn = RGB / mean(RGB) per channel.
  - Xs = 3*Rn - 2*Gn, Ys = 1.5*Rn + Gn - 1.5*Bn.
  - Bandpass Xs, Ys at 0.7-2.5 Hz.
  - alpha = std(Xs) / std(Ys), signal = Xs - alpha*Ys.
  - Hann window, overlap-add.

### POS (Wang 2017)

Locked to rPPG-Toolbox `POS_WANG`:

- Window l = ceil(1.6 * fps), stride 1.
- For each n >= l:
  - Cn = RGB[n-l:n] / mean(RGB[n-l:n]) per channel.
  - S = [[0,1,-1],[-2,1,1]] . Cn^T (2 x l).
  - h = S[0] + (std(S[0])/std(S[1])) * S[1].
  - h = h - mean(h).
  - H[n-l:n] += h (overlap-add).
- Detrend, bandpass 0.75-3 Hz, 1st-order.

### ICA (Poh, McDuff, Picard 2010)

- Detrend each of R, G, B.
- Z-score per channel.
- 3 x N matrix -> FastICA -> 3 sources.
- Pick source with highest FFT power in [0.7, 4.0] Hz.
- Bandpass 0.7-4 Hz.

FastICA: deflationary, log-cosh nonlinearity (g(u) = tanh(u)), max 200 iterations,
tol 1e-5. Self-contained, ~120 lines.

### PBV (de Haan & van Leest 2014)

- RGBn = RGB / mean(RGB) per channel (whole trace), then subtract per-channel mean.
- C = 3 x N matrix.
- Default blood-volume signature: canonical pbv = [0.33, 0.78, 0.53] (paper).
  Optional per-trace estimated mode (parameter).
- Solve W = (C . C^T)^-1 . pbv (3x3 invert), pulse = W^T . C.
- Bandpass 0.7-4 Hz.

## Shared DSP details

- **Butterworth**: bilinear-transform design, 1st and 2nd order bandpass.
  filtfilt = forward + reversed-input forward + re-reverse. Zero-phase.
- **Detrend (smoothness-prior, Tarvainen 2002)**:
  z = (I - (I + lambda^2 * D2^T . D2)^-1) . x where D2 is the 2nd-difference matrix.
  Solve via tridiagonal system (Thomas algorithm), O(N), no general matrix inverse.
- **FFT**: radix-2 Cooley-Tukey, iterative, in-place. Zero-pad to next power of 2.

## Browser adapter behavior

- `FaceRoiTracker` uses MediaPipe `FaceLandmarker`. ROI = forehead + cheeks polygons
  built from a fixed set of FaceMesh landmark indices (478-landmark model).
  Median-smoothed bbox over 5 frames. If detection fails, reuse last ROI for up to 1s.
- `frame-capture` uses `requestVideoFrameCallback` (fallback `requestAnimationFrame`).
  Per frame: draw to offscreen canvas, ROI-masked mean RGB, push `{r,g,b,t}` to a
  ring buffer (default 10s * fps).
- `LiveRppg` runs `pipeline` on the buffer every 1/updateHz seconds (default 1 Hz).
  Frame capture and HR estimation are decoupled (different timers).

Failure modes the demo surfaces:

- No camera permission -> permission-prompt UI.
- No face detected -> "Looking for face..." overlay, BPM blanked.
- Confidence < 0.3 -> BPM grey, "low confidence" label.
- First 5-10s while buffer fills -> "Calibrating..." overlay.

## Node adapter behavior

- `video-frames.ts` spawns ffmpeg with `-f rawvideo -pix_fmt rgb24`. ffmpeg is NOT
  bundled, must be on PATH (documented in README).
- `face-roi-node.ts` optional. With `@vladmandic/face-api` installed: tinyFaceDetector
  + forehead/cheek bbox heuristic (no landmark mesh). Without: full-frame mean,
  warning logged once.
- `process-video.ts` returns `{ bpmOverTime: {t, bpm}[], pulseSignal, fps }`.
  10s window, 1 Hz updates.
- CLI: `npx rppg --method pos --in video.mp4 --out result.json`.

## Testing strategy

Unit tests only (Vitest). No browser/Node-adapter integration tests in v1; the
demo page is the eyeball check.

### DSP correctness

- `fft.test.ts`: pure tone -> peak at right bin, Parseval's identity.
- `butterworth.test.ts`: out-of-band attenuation >= 30 dB, in-band < 1 dB,
  filtfilt is zero-phase (peak position unchanged).
- `detrend.test.ts`: linear trend + noise -> output mean ~ 0, slope ~ 0.
- `fast-ica.test.ts`: synthetic mixture of 3 known sources (sine, square, sawtooth);
  unmix and assert correlation > 0.95 with originals (up to sign/permutation).

### Method correctness

Synthetic RGB trace generator injects a known BPM into channels with the canonical
blood-volume ratios [0.33, 0.78, 0.53] plus per-channel additive noise.

- Per method (5): 5 BPMs (50, 70, 90, 110, 130) x 3 fps (15, 30, 60) = 75 cases.
- At SNR = 10 dB: recovered BPM within +/- 2 BPM of ground truth.
- At SNR = 0 dB: recovered BPM within +/- 5 BPM.

### Pipeline integration

- 15s synthetic trace through `pipeline()`, assert sane `bpm`, `snr`, `confidence`.

## Tooling

- TypeScript 5, strict mode, target ES2022, module ESNext.
- Vite (demo dev server + library build), Vitest (tests).
- Prettier + ESLint (typescript-eslint recommended).
- Runtime deps for core: zero.
- Peer deps: `@mediapipe/tasks-vision` (browser), `@vladmandic/face-api` (Node, optional).
- Package exports:

  ```json
  "exports": {
    ".":         "./dist/core/index.js",
    "./browser": "./dist/browser/index.js",
    "./node":    "./dist/node/index.js"
  }
  ```

- Scripts: `dev`, `build`, `test`, `test:watch`, `lint`, `format`.
- README: quickstart for both layers, method list with paper citations, MIT license,
  note that ffmpeg must be on PATH for the Node adapter.

## References

- Verkruysse, W. et al. (2008). "Remote plethysmographic imaging using ambient light." Optics Express.
- Poh, M.-Z., McDuff, D., Picard, R. (2010). "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation." Optics Express.
- de Haan, G., Jeanne, V. (2013). "Robust pulse rate from chrominance-based rPPG." IEEE TBME.
- de Haan, G., van Leest, A. (2014). "Improved motion robustness of remote-PPG by using the blood volume pulse signature." Physiol. Meas.
- Wang, W. et al. (2017). "Algorithmic principles of remote PPG." IEEE TBME.
- Tarvainen, M. P. et al. (2002). "An advanced detrending method with application to HRV analysis." IEEE TBME.
- Liu, X. et al. (2024). "rPPG-Toolbox: Deep Remote PPG Toolbox." NeurIPS Datasets & Benchmarks. https://github.com/ubicomplab/rPPG-Toolbox
