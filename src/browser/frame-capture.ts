import type { RgbSample } from '../core/types';

export type FrameCaptureOpts = {
  /** Target frames per second; sets ring-buffer capacity. Default 30. */
  targetFps?: number;
  /** Buffer length in seconds. Default 10. */
  bufferSeconds?: number;
};

type RoiLike = {
  mask: ImageData;
  bbox: { x: number; y: number; w: number; h: number };
};

type SampleListener = (s: RgbSample) => void;

// rVFC is non-standard; declare a minimal shape so `tsc` is happy.
type WithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, metadata: unknown) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

/**
 * Captures per-frame mean RGB from a video element, masked by a binary alpha
 * ROI. Uses requestVideoFrameCallback when available (fires at the actual
 * source frame rate) and falls back to requestAnimationFrame otherwise.
 *
 * Samples are pushed into a fixed-capacity ring buffer
 * (`targetFps * bufferSeconds`); listeners registered via `onSample` fire
 * synchronously after each push.
 */
export class FrameCapture {
  private readonly capacity: number;
  private buffer: RgbSample[];
  private head = 0;
  private filled = 0;
  private listeners = new Set<SampleListener>();

  private video: WithRvfc | null = null;
  private getRoi: (() => RoiLike | null) | null = null;
  private running = false;
  private rvfcHandle: number | null = null;
  private rafHandle: number | null = null;

  // Reusable offscreen canvas; sized to the source video resolution.
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(opts: FrameCaptureOpts = {}) {
    const targetFps = opts.targetFps ?? 30;
    const bufferSeconds = opts.bufferSeconds ?? 10;
    this.capacity = Math.max(1, Math.round(targetFps * bufferSeconds));
    this.buffer = new Array<RgbSample>(this.capacity);
  }

  /** Start capturing from a video element using a binary alpha mask Roi. */
  start(video: HTMLVideoElement, getRoi: () => RoiLike | null): void {
    if (this.running) return;
    this.video = video as WithRvfc;
    this.getRoi = getRoi;
    this.running = true;
    if (typeof this.video.requestVideoFrameCallback === 'function') {
      this.scheduleRvfc();
    } else {
      this.scheduleRaf();
    }
  }

  /** Stop capturing. */
  stop(): void {
    this.running = false;
    if (this.video && this.rvfcHandle !== null && this.video.cancelVideoFrameCallback) {
      try {
        this.video.cancelVideoFrameCallback(this.rvfcHandle);
      } catch {
        // ignore
      }
    }
    if (this.rafHandle !== null) {
      try {
        cancelAnimationFrame(this.rafHandle);
      } catch {
        // ignore
      }
    }
    this.rvfcHandle = null;
    this.rafHandle = null;
    this.video = null;
    this.getRoi = null;
  }

  /** Subscribe to per-frame samples. Returns an unsubscribe function. */
  onSample(cb: SampleListener): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Most recent N samples in chronological order (oldest first). */
  getRecent(n: number): RgbSample[] {
    const count = Math.min(n, this.filled);
    const out = new Array<RgbSample>(count);
    // Oldest sample lives at (head - filled) mod capacity; we copy `count`
    // samples ending at `head - 1`.
    const startIdx = (this.head - count + this.capacity) % this.capacity;
    for (let i = 0; i < count; i++) {
      out[i] = this.buffer[(startIdx + i) % this.capacity];
    }
    return out;
  }

  /** Number of samples currently buffered (saturates at capacity). */
  size(): number {
    return this.filled;
  }

  private scheduleRvfc(): void {
    if (!this.running || !this.video || !this.video.requestVideoFrameCallback) return;
    this.rvfcHandle = this.video.requestVideoFrameCallback(() => {
      if (!this.running) return;
      this.tick();
      this.scheduleRvfc();
    });
  }

  private scheduleRaf(): void {
    if (!this.running) return;
    this.rafHandle = requestAnimationFrame(() => {
      if (!this.running) return;
      this.tick();
      this.scheduleRaf();
    });
  }

  private tick(): void {
    if (!this.video || !this.getRoi) return;
    const roi = this.getRoi();
    if (!roi) return;
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (w === 0 || h === 0) return;

    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.canvas || !this.ctx) return;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.drawImage(this.video, 0, 0, w, h);

    const { bbox, mask } = roi;
    // Clamp bbox to frame just in case.
    const bx = Math.max(0, Math.min(w - 1, bbox.x));
    const by = Math.max(0, Math.min(h - 1, bbox.y));
    const bw = Math.max(1, Math.min(w - bx, bbox.w));
    const bh = Math.max(1, Math.min(h - by, bbox.h));
    const frame = this.ctx.getImageData(bx, by, bw, bh);

    // Mean RGB over pixels where the mask alpha > 0.
    const fdata = frame.data;
    const mdata = mask.data;
    // The mask was rendered at the bbox size; if it doesn't match (e.g. due to
    // a race with a bbox resize), bail.
    const sameSize = mask.width === bw && mask.height === bh;
    let rSum = 0, gSum = 0, bSum = 0, n = 0;
    if (sameSize) {
      const len = bw * bh;
      for (let i = 0; i < len; i++) {
        if (mdata[i * 4 + 3] > 0) {
          rSum += fdata[i * 4];
          gSum += fdata[i * 4 + 1];
          bSum += fdata[i * 4 + 2];
          n++;
        }
      }
    } else {
      // Fallback: just take the bbox average.
      const len = bw * bh;
      for (let i = 0; i < len; i++) {
        rSum += fdata[i * 4];
        gSum += fdata[i * 4 + 1];
        bSum += fdata[i * 4 + 2];
      }
      n = len;
    }
    if (n === 0) return;

    const sample: RgbSample = {
      r: rSum / n,
      g: gSum / n,
      b: bSum / n,
      t: performance.now(),
    };

    this.buffer[this.head] = sample;
    this.head = (this.head + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;

    for (const cb of this.listeners) cb(sample);
  }
}
