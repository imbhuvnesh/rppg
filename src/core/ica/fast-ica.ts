// Minimal FastICA for the 3-channel rPPG case (rows fixed at 3).
// Observations: array of K Float32Arrays of length N.
// Returns K independent component signals (Float32Array length N).

/**
 * Deflationary FastICA with log-cosh nonlinearity (g(u) = tanh u).
 *
 * Designed for K=3 RGB rPPG inputs but works for any K (rows = channels, columns = time).
 * Uses Jacobi-rotation eigendecomposition for whitening (suitable for small K). Random
 * initialization via `Math.random` — not seedable; expect minor cross-run variation in
 * the source ordering (sign and permutation are inherently ambiguous in ICA).
 *
 * @param obs   Observed mixed signals (length-K array of Float32Arrays, all length N).
 * @param maxIter  Max FastICA iterations per component (default 200).
 * @param tol      Convergence tolerance on |⟨w_new, w_old⟩ − 1| (default 1e-5).
 * @returns Length-K array of Float32Arrays containing the unmixed independent components.
 */
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
  // vecs[k] is the k-th eigenvector (as a row), vals[k] is its eigenvalue.
  // Row i of W_w = (1/sqrt(lambda_i)) * v_i^T, so Ww[i][j] = vecs[i][j] / sqrt(vals[i]).
  const Ww: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    Ww[i][j] = vecs[i][j] / Math.sqrt(Math.max(vals[i], 1e-10));
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
