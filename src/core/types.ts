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
