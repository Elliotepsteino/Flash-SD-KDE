
# Flash SD-KDE v2 — Kernel + Benchmark Update Plan

## 0) Scope, goals, and non-goals

### Goals
1. **Correctness first**: fix layout / chunking inconsistencies in the current 16D empirical-score path; add stability clamps.
2. **Performance**: reduce/avoid global atomics where they bottleneck; reduce redundant query loads; add autotuning.
3. **Reproducible evidence**: add a real, low-friction benchmark with quantitative metrics + compelling qualitative artifacts.

### Explicit non-goals (for this update)
- **No RFF / Random Fourier Features** (changes estimator and requires more kernel + plumbing).
- **No multi-step SD-KDE** (iterated correction changes algorithmic surface and evaluation story).

---

## 1) Ground-truth definition of the algorithm (for implementation sanity)

We implement **one-step SD-KDE**:

1. Compute a score estimate at data points: \(\hat s(x_i)\).
2. Shift data points: \(\tilde x_i = x_i + \delta\,\hat s(x_i)\).
3. Evaluate a KDE with kernel bandwidth \(h\) on the shifted set: \(\hat p(x)=\frac{1}{n h^d}\sum_i K((x-\tilde x_i)/h)\).

When using the *empirical* score for Gaussian KDE in d dimensions, a convenient identity is:

- \(\nabla \log \hat p(x_i) = \frac{1}{h^2}\left(\frac{\sum_j \phi_{ij} x_j}{\sum_j \phi_{ij}} - x_i\right)\)
- \(\phi_{ij} = \exp(-\|x_i-x_j\|^2/(2h^2))\)

We keep \(\delta = h^2/2\) as the default (the theoretically motivated choice).

---

## 2) Critical correctness fixes (must land before perf work)

### 2.1 Standardize the memory layout of `weighted_sum`
The current repo mixes feature-major and row-major layouts for the 16D empirical score accumulator and slices it inconsistently when chunking. v2 will standardize everything to:

- `weighted_sum`: shape **(n_query, 16)**, contiguous row-major float32
- `pdf_sum`: shape **(n_query,)**, float32

**Why this layout:**
- Matches normal PyTorch tensors and downstream math: `ratio = weighted_sum / (pdf_sum[:, None] + eps)`.
- Makes chunking natural: `weighted_sum[q0:q1]` slices by queries.

**Acceptance checks:**
- For small n (e.g., 256–2048), compare against a PyTorch reference for:
  - `pdf_sum` and `weighted_sum` (absolute + relative error)
  - derived score `(ratio - x)/h^2`
- Chunked execution equals unchunked execution.

### 2.2 Clamp tiny negative squared distances in *all* dot-based kernels
Whenever using \(\|q-x\|^2 = \|q\|^2 + \|x\|^2 - 2 q^T x\), numerical rounding can make the result slightly negative.

Add:
- `dist = tl.maximum(dist, 0.0)`

in **every** kernel that computes `dist` via dot identity (KDE eval and empirical-score kernels).

### 2.3 Add an explicit precision mode
Expose:
- `precision_mode='fast_tf32'`: allow TF32 tensor core math in dot products
- `precision_mode='fp32_ieee'`: force IEEE-ish FP32 dot behavior for validation runs

This makes it easy to debug numerical deltas and produce “accuracy vs speed” plots.

---

## 3) Performance upgrades (kernel designs)

### Overview: why v2 changes the structure
The current kernels rely heavily on a 2D grid (query blocks × data blocks) and `atomic_add` to accumulate partials. That is simple, but can bottleneck due to:
- contention/serialization on atomics
- redundant query loads (each query tile is reloaded once per data tile)

v2 introduces two main ideas:

1. **Split-K two-pass reduction (exact):**
   - pass A: compute partial sums for a fixed “split” of the data (no atomics)
   - pass B: reduce partials across splits

2. **Streaming (“Flash-style”) inner loops:**
   - load queries once per program
   - stream through multiple data chunks inside the same program
   - avoid materializing large intermediates (like full `[BLOCK_M × BLOCK_N]` phi)

---

## 4) Split-K building block (used everywhere)

### 4.1 Definitions
Choose fixed compile-time parameters:
- `BLOCK_M`: queries per program (e.g., 32/64/128)
- `BLOCK_N_ITER`: points per inner iteration (e.g., 64/128/256)
- `ITERS_PER_SPLIT`: number of iterations per program (e.g., 4/8/16/32)

Then each program covers:
- `BLOCK_N_TOTAL = BLOCK_N_ITER * ITERS_PER_SPLIT` data points

Number of splits:
- `n_splits = ceil(n_data / BLOCK_N_TOTAL)`

### 4.2 Output shapes for partials
- KDE eval partials: `partial_pdf`: shape `(n_splits, n_query)`
- Emp score ordered partials:
  - `partial_pdf`: shape `(n_splits, n_query)`
  - `partial_weighted`: shape `(n_splits, n_query, 16)`

Then pass B reduces along `split` dimension.

### 4.3 Memory planning
Partial buffers can be large. Use heuristics:
- Prefer large `BLOCK_N_TOTAL` to keep `n_splits` small.
- For empirical score, cap `n_splits` so that `partial_weighted` fits comfortably.

Rule of thumb memory:
- `partial_weighted_bytes ≈ n_splits * n_query * 16 * 4`

Example: `n_query=32768`, `n_splits=8` ⇒ ~ 16 MB for weighted partials (plus pdf partials) — reasonable.

---

## 5) KDE evaluation kernels (1D + 16D)

### 5.1 1D KDE eval: Split-K + reduction
**Pass A:** each program handles `(query_block, split_id)`
- load `q[BLOCK_M]` once
- stream over `ITERS_PER_SPLIT` data chunks
- compute contributions `exp(-(q-x)^2/(2h^2))` and accumulate into registers
- write `partial_pdf[split_id, q_indices]`

**Pass B:** reduce `partial_pdf[:, q]` across `split`.

**Result:** exact, deterministic, and no atomics.

### 5.2 16D KDE eval: streaming + Split-K
Implement a streaming kernel for `d=16`:
- load `Q_block (BLOCK_M × 16)` once
- optionally load `Q_norms`
- for each inner iter:
  - load `X_chunk (BLOCK_N_ITER × 16)`
  - compute `q·x` via `tl.dot` (TF32 optional)
  - compute `dist = q_norm + x_norm - 2 qx` and clamp
  - compute `phi = exp(-0.5*dist/h^2)`
  - reduce across N to get per-query partial sum
- write to `partial_pdf`

**Pass B:** reduce partials across splits.

---

## 6) Empirical score kernels (16D)

We implement two backends and choose via heuristic:

1. **Ordered Split-K (deterministic, no atomics):** compute all ordered pairs.
2. **Symmetric upper-triangular (≈2× less compute, uses atomics):** compute only block pairs with `bi <= bj` and update both blocks.

### 6.1 Ordered Split-K backend (baseline + correctness reference)
**Pass A:** for each `(query_block, split_id)`
- load `Q_block = X[q_idx]` (since queries are data)
- stream over a slice of X
- accumulate:
  - `pdf_partial[q] += sum_j phi(q,j)`
  - `weighted_partial[q,:] += sum_j phi(q,j) * X[j,:]`
- write `partial_pdf` and `partial_weighted`

**Pass B:** reduce partials across splits to global `pdf_sum` and `weighted_sum`.

**Why we want it:**
- deterministic and robust
- excellent for validating the symmetric kernel

### 6.2 Symmetric backend (upper triangular block pairs)
Let `B` be the block size (e.g., 64). Let `nb = ceil(n/B)`.

We run programs over block-pairs `(bi, bj)` with `bi <= bj`.

#### Off-diagonal case (`bi < bj`)
Load:
- `Xi = X[bi]` (B×16)
- `Xj = X[bj]` (B×16)

Compute `phi` (B×B).

Accumulate:
- For block i:
  - `pdf_i += row_sum(phi)`
  - `w_i += phi @ Xj`
- For block j:
  - `pdf_j += col_sum(phi)`
  - `w_j += phi^T @ Xi`

Use `atomic_add` to global `pdf_sum` and `weighted_sum` for both blocks.

#### Diagonal case (`bi == bj`)
Compute only the **upper triangle** of `phi` (including diagonal) via mask `i_local <= j_local`.

To add each pair exactly once:
- `pdf += row_sum(phi_upper) + col_sum(phi_upper) - diag(phi_upper)`
- `w += (phi_upper @ Xb) + (phi_upper^T @ Xb) - diag(phi_upper)[:,None] * Xb`

Then atomic-add into global buffers.

#### Notes
- This kernel reduces compute ~2× (dominant part) compared to ordered.
- Atomics remain, but total work drops. Often this is a large net win.

### 6.3 Backend selection heuristic
Start simple:
- If `n <= N_SYM_MAX` (e.g., 50k) ⇒ use `symmetric_atomic` (fastest in many cases)
- Else ⇒ use `ordered_splitk` (safer memory/perf profile)

Also allow manual override.

---

## 7) Norm precompute (exact, optional)

Distance uses `||q||^2` and `||x||^2`. Precomputing norms can reduce compute and register pressure.

Add optional inputs:
- `data_norms: (n_data,)`
- `query_norms: (n_query,)` (or reuse `data_norms` for empirical score)

Wrapper policy:
- Enable by default for 16D paths.
- Keep optional for 1D.

---

## 8) Autotuning and measurement

### 8.1 Autotune targets
Autotune per kernel family over:
- `BLOCK_M ∈ {32, 64, 128}`
- `BLOCK_N_ITER ∈ {64, 128, 256}`
- `ITERS_PER_SPLIT ∈ {4, 8, 16, 32}`
- `num_warps ∈ {2, 4, 8}`
- `num_stages ∈ {2, 3, 4}`

Separate autotune tables for:
- 1D KDE eval (Split-K)
- 16D KDE eval (streaming Split-K)
- 16D empirical score ordered Split-K
- 16D empirical score symmetric atomic

### 8.2 Benchmark sizes for tuning
Use representative regimes:
- KDE eval: `(n_data, n_query)` in {(8k,8k), (32k,8k), (64k,8k), (64k,64k)}
- Emp score self: `n in {4k, 8k, 16k, 32k}`

Log for each config:
- runtime (ms)
- interactions/s = `n_data*n_query / time`
- peak workspace bytes
- precision mode

---

## 9) Public Python API changes (wrappers)

Expose these knobs (with good defaults):

- `precision_mode`: `'fast_tf32' | 'fp32_ieee'`
- `kde_backend`: `'splitk_stream' | 'atomic'` (atomic kept for debugging)
- `emp_score_backend`: `'symmetric_atomic' | 'ordered_splitk'`
- `use_precomputed_norms: bool`
- `autotune: bool` (or always-on with cached best config)

Provide a clean end-to-end function:
- `emp_sd_kde_fit_transform(X, h, ...) -> X_debiased`
- `kde_eval(X_train_or_X_debiased, Q, h, ...) -> densities`

---

## 10) Validation plan (correctness + stability)

### 10.1 Unit tests (small sizes)
For `n_query,n_data <= 2048`:
- Compare KDE eval vs a PyTorch reference implementation.
- Compare empirical score (`pdf_sum`, `weighted_sum`, `score`) vs reference.
- Verify chunked vs unchunked equality.

### 10.2 Cross-backend consistency
- Compare `ordered_splitk` empirical score vs `symmetric_atomic` empirical score (same h, same precision mode) on moderate n (e.g., 4096). Differences should be small (mostly numerical ordering).

### 10.3 Numeric stability
- Confirm `dist` clamp prevents `exp(+epsilon) > 1`.
- Guard divisions with `eps` for `pdf_sum`.
- Confirm no NaNs/inf across kernels for typical ranges.

---

# 11) Real benchmark: MNIST vs Fashion-MNIST OOD detection in PCA-16 space

This is a “real data, easy to run, visually compelling” demo that aligns perfectly with the 16D specialization.

## 11.1 Dataset and representation
- ID: MNIST train (fit), MNIST test (evaluate)
- OOD: Fashion-MNIST test (evaluate)

Steps:
1. Load MNIST train/test + Fashion-MNIST test (torchvision).
2. Flatten images to 784 floats, normalize consistently.
3. Fit PCA with `n_components=16` on MNIST train.
4. Transform:
   - `X_train_16` (MNIST train)
   - `X_id_16` (MNIST test)
   - `X_ood_16` (Fashion test)

## 11.2 Methods compared
1. **KDE**: Gaussian KDE on `X_train_16`, evaluate densities on ID and OOD.
2. **Emp-SD-KDE**:
   - empirical score on `X_train_16`
   - shift with `delta = h^2/2`
   - KDE on debiased set, evaluate densities on ID and OOD

## 11.3 Bandwidth selection
Start with your existing multivariate Silverman/Scott heuristic.
Optional refinement (recommended):
- Split MNIST train into train/val (e.g., 50k/10k).
- Choose `h` by maximizing mean log-likelihood on val over multipliers:
  - `h ∈ {0.5, 0.75, 1.0, 1.25, 1.5} * h_silverman`
Use the selected `h` for both methods for fairness.

## 11.4 Metrics
OOD detection:
- ROC AUC (primary)
- PR AUC (secondary)

Density quality:
- Mean log-likelihood on MNIST test (ID)

Runtime:
- Wall-clock breakdown:
  - `T_score` (Emp-SD-KDE only)
  - `T_shift`
  - `T_eval_id`, `T_eval_ood`
- Also record kernel-only timings (CUDA events).

## 11.5 Sweeps for scaling plots
Subsample MNIST train to:
- `n_train ∈ {2k, 4k, 8k, 16k, 32k}`

For each n:
- compute AUCs
- record runtimes

## 11.6 Plots and qualitative artifacts
Generate and save:

1. **ROC curves** (KDE vs Emp-SD-KDE)
2. **Histogram of log densities** for ID vs OOD (overlay KDE and Emp-SD-KDE)
3. **AUC vs n_train** (log-scale x)
4. **Runtime vs n_train** with breakdown bars
5. **Image grids** (the “demo candy”):
   - MNIST test: top-25 highest density
   - MNIST test: top-25 lowest density
   - Fashion test: top-25 *highest* MNIST-density (false positives)

---

## 12) Repo deliverables (minimal file list)

### Kernels
- `kernels/kde_eval_1d_splitk.py` (or integrated in existing triton file)
- `kernels/kde_eval_16d_stream_splitk.py`
- `kernels/emp_score_16d_ordered_splitk.py`
- `kernels/emp_score_16d_symmetric_atomic.py`
- `kernels/reduce_partials.py` (generic reduction for partial buffers)

### Wrappers / API
- `flash_sd_kde/kde.py` (public wrappers, backend selection, norms option)
- `flash_sd_kde/config.py` (autotune caches + heuristics)

### Benchmarks
- `benchmarks/mnist_fashion_pca16_ood.py` (end-to-end run; saves JSON)
- `plots/plot_mnist_fashion_ood.py` (reads JSON; writes PDFs/PNGs)
- `plots/save_density_ranked_grids.py` (writes image grids)

---

## 13) Milestones (suggested sequencing)

1. **Correctness stabilization**: fix `weighted_sum` layout, chunking, add dist clamp, add precision mode.
2. **KDE eval Split-K**: 1D then 16D streaming Split-K; verify vs reference.
3. **Emp score ordered Split-K**: implement + validate + profile memory.
4. **Symmetric empirical score**: implement upper-triangular + diagonal handling; validate vs ordered.
5. **Autotune + heuristics**: add autotune configs; choose defaults.
6. **Benchmark + plots**: MNIST/Fashion PCA-16 end-to-end; produce plots and artifacts.

---

## 14) Definition of success

- **Correctness:** passes reference tests on small sizes; chunked == unchunked; no NaNs/inf; consistent across backends.
- **Performance:**
  - KDE eval is faster than baseline atomic version for realistic sizes.
  - Empirical score stage shows clear speedup (ideally ~2× from symmetry vs ordered; and/or material improvement from Split-K vs atomics).
- **Evidence:** a single benchmark script produces a report folder with:
  - AUC / log-likelihood numbers
  - ROC + scaling plots
  - density-ranked image grids
  - runtime breakdown
