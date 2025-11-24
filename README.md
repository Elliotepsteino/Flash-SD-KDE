# Flash-SD-KDE

This repo now contains a Triton implementation of the standard Gaussian KDE so
we can benchmark it against the existing Silverman estimator.

- `triton_kde.py` provides the GPU kernel plus a torch wrapper.
- `kde_utils.py` exposes the mixture definitions, samplers, and CPU KDE helper
  functions (keeps benchmarks lightweight).
- `benchmark_triton_kde.py` samples 5000 training points and 500 test points
  for 3 seeds (configurable) and reports CPU Silverman, scikit-learn, GPU Silverman,
  and GPU SD-KDE timings plus accuracy deltas.
- `run_sweep.sh` plus `plot_flash_sd_kde.py` help automate the
  power-of-two sweeps and produce a PDF plot for papers.
- `report/main.tex` is a concise \LaTeX{} report that includes the speedup plot.

## Running the benchmark

1. Activate the conda env that has PyTorch, Triton, NumPy, SciPy:
   ```bash
   conda activate flash-sd-kde
   ```
2. Run the benchmark (replace the device if needed):
   ```bash
   python benchmark_triton_kde.py --seeds 0,1,2 --n-train 32768 --n-test 4096 --mixture-index 0 --device cuda
   ```
   - Add `--cpu-only` if CUDA isn’t available and you just want to test the CPU
     path, `--skip-sklearn` to skip the scikit-learn baseline, or
     `--skip-emp-gpu` to skip the GPU empirical SD-KDE run.
   - Add `--emp-kernel-only` to benchmark only the Triton SD-KDE kernel (useful
     for profiling with Nsight tools).
   - Use `--mixture-index 0|1|2` to select which mixture from
     `kde_utils.py` you want to evaluate.

## Automated sweep → PDF plot

1. Run the sweep (keeps the 8:1 ratio by setting `n_test = n_train / 8`):
   ```bash
   chmod +x run_sweep.sh  # once per checkout
   ./run_sweep.sh sweep.log
   ```
   - Override seeds/mixture/device with `SEEDS`, `MIXTURE`, `DEVICE` env vars.
2. Convert the resulting log into `flash-sd-kde.pdf` (sklearn vs SD-KDE bars):
   ```bash
   python plot_flash_sd_kde.py --log sweep.log --output flash-sd-kde.pdf
   ```
   The generated figure uses bars for sklearn, Triton SD-KDE, and
   Torch SD-KDE GPU runtimes at each `n_train` value (1,024 → 65,536,
   doubling), and annotates the Triton empirical bar with its speedup over
   both sklearn and Torch.

3. Optionally, generate a utilization plot (Triton vs Torch SD-KDE):
   ```bash
   python plot_emp_sd_kde_util.py --log sweep.log --output emp-sd-kde-util.pdf
   ```
   This plot uses a simple flop model to estimate FLOPs for SD-KDE
   at each `n_train`, divides by the measured runtimes, and reports GPU
   utilization as a percentage of the A6000 FP32 peak.

4. Compile the report (from the `report` directory):
   ```bash
   cd report
   pdflatex main.tex
   ```

## Large-n Triton-only scaling sweep

To stress only the Triton empirical SD-KDE kernel (score + shift + KDE) on
multi-million-sample problems, use the helper script below.  It sweeps powers
of two from $2^{15}$ up to $2^{22}$ (32{,}768 → 4{,}194{,}304) with a single
seed, fixing $n_{\text{test}} = n_{\text{train}}/8$ in each case:

```bash
chmod +x run_triton_scaling.sh  # once per checkout
./run_triton_scaling.sh large_triton.log
```

Override the defaults with environment variables:

```bash
SEED=1 DEVICE=cuda:1 START_POWER=17 END_POWER=22 ./run_triton_scaling.sh
```

The script calls `benchmark_triton_kde.py --emp-kernel-only ...` so only the
Triton kernels execute; the resulting log pairs well with `nsys profile` for
collecting timeline data on the largest configurations.  To visualize the
achieved utilization, feed the log into the helper plotter:

```bash
python plot_triton_large_util.py --log large_triton.log --output triton-large-util.pdf
```

## 16-D tensor-core KDE

We also support a fixed-width (16 dimensional) Gaussian KDE that keeps the 1-D
code paths intact while showcasing how the problem maps to Tensor Cores.  A
pairwise squared distance between $x_i, y_j \in \\mathbb{R}^{16}$ decomposes as
$\lVert x_i\rVert^2 + \lVert y_j\rVert^2 - 2 x_i^\top y_j$, so the dominant
cost is the dot-product matrix $XY^\top$.  For $d=16$ this becomes a small
GEMM that we execute via Triton's `tl.dot` interface, enabling Tensor Cores to
handle $>90\\%$ of the FLOPs.  Vector norms, broadcasted additions, and the
final exponential are still $O(n_{\\text{train}} n_{\\text{test}})$ but are
minor once the GEMM is accelerated.

Run the new benchmark with:

```bash
python benchmark_triton_kde.py \
  --multi-d \
  --seeds 0,1 \
  --n-train 65536 \
  --n-test 8192 \
  --device cuda
```

This routine samples a simple isotropic two-component Gaussian mixture in
16 dimensions, computes a Silverman-style bandwidth, compares Triton against
scikit-learn's `KernelDensity`, and reports accuracy plus end-to-end speedup.
Pass `--multi-d-bandwidth <value>` to override the adaptive bandwidth.

The empirical SD-KDE step also benefits from matrix-multiply structure.  Its
score numerator involves terms of the form
$\sum_j -(x_i - y_j)\,\varphi_{ij}$ with $\varphi_{ij} = \exp(-\lVert
x_i - y_j \rVert^2 / (2h^2))$.  Using the identity
$\sum_j (x_i - y_j)\varphi_{ij} = x_i \sum_j \varphi_{ij} - \sum_j \varphi_{ij}
y_j$, both the KDE evaluation and the score numerator reduce to GEMMs:
the dot-product matrix $XY^\top$ yields squared distances, while
$\Phi Y$ (with $\Phi_{ij} = \varphi_{ij}$) produces the weighted sums needed
for the numerator.  Triton maps both GEMMs to Tensor Cores, whereas the Torch
baseline relies on standard SIMT FP32 kernels.

To benchmark SD-KDE (score+shift+KDE) directly, add `--multi-d-sd`:

```bash
python benchmark_triton_kde.py \
  --multi-d-sd \
  --seeds 0,1 \
  --n-train 32768 \
  --n-test 4096 \
  --device cuda
```

This run compares a Torch GEMM-based SD-KDE pipeline against the Triton
Tensor-Core implementation (which reduces both the KDE and score numerators to
blocked GEMMs) and prints accuracy deltas plus runtime ratios.
Add `--sd-nd-triton-only` to skip the Torch baseline when you just need Triton
timings (useful for profiling or long sweeps).

To sweep multiple sizes automatically (single seed), use the helper script:

```bash
chmod +x run_triton_sd_kde_nd.sh  # once per checkout
./run_triton_sd_kde_nd.sh triton_sd_kde_nd.log
```

Adjust the range or seed via `START_POWER`, `END_POWER`, and `SEED` environment
variables (e.g. `SEED=1 START_POWER=13 END_POWER=18 ./run_triton_sd_kde_nd.sh`).
The script launches only the Triton path (Torch is skipped) so the log is ideal
for profiling or utilization analysis.
Once you have the log you can convert it into a utilization plot (Torch vs
Triton) with:

```bash
python plot_triton_sd_kde_nd_util.py \
  --log triton_sd_kde_nd.log \
  --output triton-sd-kde-nd-util.pdf
```

To compare runtimes for sklearn KDE, SD-KDE (Torch), and SD-KDE (Triton) up to
32k samples, run the combined sweep and plot:

```bash
chmod +x run_nd_runtime_sweep.sh  # once per checkout
./run_nd_runtime_sweep.sh nd_runtime.log
python plot_nd_runtime.py --log nd_runtime.log --output nd-runtime.pdf
```

## Profiling the Triton SD-KDE kernel

Use the `--emp-kernel-only` flag to time just the Triton SD-KDE kernel
(score + shift + final KDE). This is handy when profiling with Nsight tools:

```bash
python benchmark_triton_kde.py \
  --seeds 0,1,2 \
  --n-train 32768 \
  --n-test 4096 \
  --mixture-index 0 \
  --device cuda \
  --emp-kernel-only
```

To capture a timeline without requiring perf-counter access:

```bash
nsys profile --force-overwrite true -o sd-kde --trace=cuda \
  python benchmark_triton_kde.py \
    --seeds 0 \
    --n-train 32768 \
    --n-test 4096 \
    --mixture-index 0 \
    --device cuda \
    --emp-kernel-only
nsys stats --force-export true sd-kde.nsys-rep
```

For the 16-D SD-KDE kernel, replace the command with:

```bash
nsys profile --force-overwrite true -o sd-kde-nd --trace=cuda \
  python benchmark_triton_kde.py \
    --multi-d-sd \
    --sd-nd-triton-only \
    --seeds 0 \
    --n-train 32768 \
    --n-test 4096 \
    --device cuda
nsys stats --force-export true sd-kde-nd.nsys-rep
```

The resulting `.nsys-rep` captures both the tensor-core KDE GEMM and the SD-KDE
score GEMM, mirroring the 1-D profiling workflow but for the ND kernels.
