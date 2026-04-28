import { pipeline } from '../core/pipeline';
import type { MethodName, RppgResult, RgbSample, RgbTrace } from '../core/types';
import { FaceRoiTracker, type Roi } from './face-roi';
import { FrameCapture } from './frame-capture';

export type LiveRppgOpts = {
  method: MethodName;
  /** HR-estimation window length in seconds. Default 10. */
  windowSec?: number;
  /** Pipeline update rate in Hz. Default 1. */
  updateHz?: number;
  /** Override for the FaceLandmarker model URL. */
  modelUrl?: string;
  /** Override for the MediaPipe WASM base URL (e.g. self-hosted/offline). */
  wasmBase?: string;
};

const MIN_CALIBRATION_SEC = 5;

type ResultListener = (r: RppgResult) => void;

/**
 * Composes FaceRoiTracker + FrameCapture + the rPPG pipeline into a live HR
 * estimator. Frame capture runs at the video frame rate; the pipeline runs on
 * a setInterval at `updateHz`, so the math doesn't recompute per frame.
 */
export class LiveRppg {
  private readonly opts: Required<Omit<LiveRppgOpts, 'modelUrl' | 'wasmBase'>> & {
    modelUrl?: string;
    wasmBase?: string;
  };
  private tracker = new FaceRoiTracker();
  private capture: FrameCapture;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private listeners = new Set<ResultListener>();
  private lastResult: RppgResult | null = null;
  private started = false;

  constructor(opts: LiveRppgOpts) {
    this.opts = {
      method: opts.method,
      windowSec: opts.windowSec ?? 10,
      updateHz: opts.updateHz ?? 1,
      modelUrl: opts.modelUrl,
      wasmBase: opts.wasmBase,
    };
    // Capture buffer needs to comfortably hold the analysis window.
    this.capture = new FrameCapture({
      targetFps: 30,
      bufferSeconds: Math.max(this.opts.windowSec, 10),
    });
  }

  async start(video: HTMLVideoElement): Promise<void> {
    if (this.started) return;
    this.started = true;
    await this.tracker.init(this.opts.modelUrl, this.opts.wasmBase);
    this.capture.start(video, () => this.tracker.detect(video));
    const periodMs = 1000 / this.opts.updateHz;
    this.intervalId = setInterval(() => this.tickPipeline(), periodMs);
  }

  stop(): void {
    if (!this.started) return;
    this.started = false;
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.capture.stop();
    this.tracker.dispose();
  }

  onUpdate(cb: ResultListener): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Last known result, or null if not yet computed. */
  latest(): RppgResult | null {
    return this.lastResult;
  }

  /** Last detected ROI from the tracker, or null if none yet. */
  getRoi(): Roi | null {
    return this.tracker.getLastRoi();
  }

  private tickPipeline(): void {
    const desired = Math.ceil(this.opts.windowSec * 60); // generous over-estimate
    const samples = this.capture.getRecent(desired);
    if (samples.length < 2) return;
    const span = (samples[samples.length - 1].t - samples[0].t) / 1000;
    if (span < MIN_CALIBRATION_SEC) return;

    // Trim to the most recent windowSec of samples (timestamps are monotonic).
    const cutoff = samples[samples.length - 1].t - this.opts.windowSec * 1000;
    let firstIdx = 0;
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].t >= cutoff) {
        firstIdx = i;
        break;
      }
    }
    const window = samples.slice(firstIdx);
    if (window.length < 2) return;

    const trace = toRgbTrace(window);
    if (!Number.isFinite(trace.fps) || trace.fps <= 0) return;
    const result = pipeline(trace, this.opts.method);
    this.lastResult = result;
    for (const cb of this.listeners) cb(result);
  }
}

function toRgbTrace(samples: RgbSample[]): RgbTrace {
  const N = samples.length;
  const r = new Float32Array(N);
  const g = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    r[i] = samples[i].r;
    g[i] = samples[i].g;
    b[i] = samples[i].b;
  }
  // Observed fps from timestamps (ms). Matters when frames drop.
  const dtMs = samples[N - 1].t - samples[0].t;
  const fps = dtMs > 0 ? ((N - 1) / dtMs) * 1000 : 0;
  return { r, g, b, fps };
}
