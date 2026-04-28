// Wraps MediaPipe FaceLandmarker to compute a forehead+cheeks ROI mask.
// MediaPipe is loaded lazily so the core remains zero-dep at module load.

export type Roi = {
  /** Binary alpha mask covering the bbox region (alpha 255 = inside ROI). */
  mask: ImageData;
  /** Bounding box of the ROI in pixel coordinates of the source video frame. */
  bbox: { x: number; y: number; w: number; h: number };
  /** Full set of normalized landmarks (0..1) returned by FaceLandmarker. */
  landmarks: { x: number; y: number }[];
};

// FaceMesh landmark indices for forehead + cheek polygons.
const FOREHEAD_IDX = [10, 109, 67, 103, 54, 21, 162, 127, 234];
const LEFT_CHEEK_IDX = [50, 101, 36, 205, 187, 123, 116];
const RIGHT_CHEEK_IDX = [280, 330, 266, 425, 411, 352, 345];

const DEFAULT_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// Keep in sync with @mediapipe/tasks-vision in package.json. The WASM glue
// and JS API are versioned together, so a mismatch causes runtime failures.
const DEFAULT_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

const SMOOTH_FRAMES = 5;
const REUSE_TIMEOUT_MS = 1000;

type Landmark = { x: number; y: number; z: number };
type FaceLandmarkerLike = {
  detectForVideo(video: HTMLVideoElement, timestamp: number): {
    faceLandmarks: Landmark[][];
  };
  close(): void;
};

type Bbox = { x: number; y: number; w: number; h: number };

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function fillPolygon(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  offsetX: number,
  offsetY: number
): void {
  if (points.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x - offsetX, points[0].y - offsetY);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x - offsetX, points[i].y - offsetY);
  }
  ctx.closePath();
  ctx.fill();
}

export class FaceRoiTracker {
  private landmarker: FaceLandmarkerLike | null = null;
  private maskCanvas: HTMLCanvasElement | null = null;
  private maskCtx: CanvasRenderingContext2D | null = null;
  private bboxHistory: Bbox[] = [];
  private lastRoi: Roi | null = null;
  private lastDetectionMs = 0;

  /** Lazy-load MediaPipe and the model. */
  async init(modelUrl?: string, wasmBase?: string): Promise<void> {
    if (this.landmarker) return;
    const vision = await import('@mediapipe/tasks-vision');
    const fileset = await vision.FilesetResolver.forVisionTasks(wasmBase ?? DEFAULT_WASM_BASE);
    this.landmarker = (await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl ?? DEFAULT_MODEL_URL },
      runningMode: 'VIDEO',
      numFaces: 1,
    })) as unknown as FaceLandmarkerLike;
  }

  /** Last detected ROI (or last cached one within the reuse window), if any. */
  getLastRoi(): Roi | null {
    return this.lastRoi;
  }

  /** Detect face in current video frame; returns ROI or null. */
  detect(video: HTMLVideoElement): Roi | null {
    if (!this.landmarker) return null;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w === 0 || h === 0) return this.reuseLastIfFresh();

    const tsMs = performance.now();
    let result: { faceLandmarks: Landmark[][] };
    try {
      result = this.landmarker.detectForVideo(video, tsMs);
    } catch {
      return this.reuseLastIfFresh();
    }
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) return this.reuseLastIfFresh();

    const lm = faces[0];
    const toPx = (idx: number) => ({ x: lm[idx].x * w, y: lm[idx].y * h });
    const forehead = FOREHEAD_IDX.map(toPx);
    const leftCheek = LEFT_CHEEK_IDX.map(toPx);
    const rightCheek = RIGHT_CHEEK_IDX.map(toPx);

    // Compute raw bbox over all polygon vertices.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of [...forehead, ...leftCheek, ...rightCheek]) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const rawBbox: Bbox = {
      x: Math.max(0, Math.floor(minX)),
      y: Math.max(0, Math.floor(minY)),
      w: Math.min(w, Math.ceil(maxX)) - Math.max(0, Math.floor(minX)),
      h: Math.min(h, Math.ceil(maxY)) - Math.max(0, Math.floor(minY)),
    };
    if (rawBbox.w <= 0 || rawBbox.h <= 0) return this.reuseLastIfFresh();

    // Median-smooth bbox over the last SMOOTH_FRAMES frames.
    this.bboxHistory.push(rawBbox);
    if (this.bboxHistory.length > SMOOTH_FRAMES) this.bboxHistory.shift();
    const smoothBbox: Bbox = {
      x: Math.round(median(this.bboxHistory.map((b) => b.x))),
      y: Math.round(median(this.bboxHistory.map((b) => b.y))),
      w: Math.round(median(this.bboxHistory.map((b) => b.w))),
      h: Math.round(median(this.bboxHistory.map((b) => b.h))),
    };
    // Clamp smoothed bbox to frame.
    smoothBbox.x = Math.max(0, smoothBbox.x);
    smoothBbox.y = Math.max(0, smoothBbox.y);
    smoothBbox.w = Math.max(1, Math.min(w - smoothBbox.x, smoothBbox.w));
    smoothBbox.h = Math.max(1, Math.min(h - smoothBbox.y, smoothBbox.h));

    // Render polygons to an offscreen canvas at bbox size.
    if (!this.maskCanvas) {
      this.maskCanvas = document.createElement('canvas');
      this.maskCtx = this.maskCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.maskCanvas || !this.maskCtx) return this.reuseLastIfFresh();
    if (this.maskCanvas.width !== smoothBbox.w || this.maskCanvas.height !== smoothBbox.h) {
      this.maskCanvas.width = smoothBbox.w;
      this.maskCanvas.height = smoothBbox.h;
    }
    const ctx = this.maskCtx;
    ctx.clearRect(0, 0, smoothBbox.w, smoothBbox.h);
    ctx.fillStyle = 'rgba(255,255,255,1)';
    fillPolygon(ctx, forehead, smoothBbox.x, smoothBbox.y);
    fillPolygon(ctx, leftCheek, smoothBbox.x, smoothBbox.y);
    fillPolygon(ctx, rightCheek, smoothBbox.x, smoothBbox.y);
    const mask = ctx.getImageData(0, 0, smoothBbox.w, smoothBbox.h);

    const roi: Roi = {
      mask,
      bbox: smoothBbox,
      landmarks: lm.map((p) => ({ x: p.x, y: p.y })),
    };
    this.lastRoi = roi;
    this.lastDetectionMs = tsMs;
    return roi;
  }

  /** Free MediaPipe resources. */
  dispose(): void {
    if (this.landmarker) {
      try {
        this.landmarker.close();
      } catch {
        // ignore
      }
      this.landmarker = null;
    }
    this.bboxHistory = [];
    this.lastRoi = null;
    this.maskCanvas = null;
    this.maskCtx = null;
  }

  private reuseLastIfFresh(): Roi | null {
    if (this.lastRoi && performance.now() - this.lastDetectionMs <= REUSE_TIMEOUT_MS) {
      return this.lastRoi;
    }
    return null;
  }
}
