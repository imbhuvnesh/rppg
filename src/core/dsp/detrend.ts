// Tarvainen 2002 smoothness-prior detrending.
// z_stationary = x - (I + lambda^2 D2^T D2)^-1 x
// D2 is the (N-2) x N second-difference matrix: each row [..1 -2 1..].
// (I + lambda^2 D2^T D2) is symmetric pentadiagonal: bands -2,-1,0,1,2.
// We build the 5 diagonals and run a banded LDL^T solve (no pivoting).

/**
 * Smoothness-prior detrending (Tarvainen 2002). Subtracts a smooth trend
 * obtained by solving (I + λ² D₂ᵀ D₂) z = x and returning x - z. Larger λ
 * means smoother trend (more high-frequency content kept in the residual).
 * Effective high-pass cutoff: fc ≈ (fs/π)·sqrt((1/(2λ))·(1+sqrt(1+4λ²))).
 * λ=100 puts the cutoff at ~0.04 cycles/sample, suitable for HR signals at
 * fs ≥ 15 Hz.
 */
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
      const v = a2[i];
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
