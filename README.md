# Flash-SD-KDE

This repo now contains a Triton implementation of the standard Gaussian KDE so
we can benchmark it against the existing Silverman estimator.

- `triton_kde.py` provides the GPU kernel plus a torch wrapper.
- `kde_utils.py` exposes the mixture definitions, samplers, and CPU KDE helper
  functions (keeps benchmarks lightweight).
- `benchmark_triton_kde.py` samples 5000 training points and 500 test points
  for 3 seeds (configurable) and reports CPU Silverman, scikit-learn, GPU Silverman,
  and GPU empirical SD-KDE timings plus accuracy deltas.

## Running the benchmark

1. Activate the conda env that has PyTorch, Triton, NumPy, SciPy:
   ```bash
   conda activate flash-sd-kde
   ```
2. Run the benchmark (replace the device if needed):
   ```bash
   python benchmark_triton_kde.py --seeds 0,1,2 --n-train 5000 --n-test 500 --mixture-index 0 --device cuda
   ```
   - Add `--cpu-only` if CUDA isn’t available and you just want to test the CPU
     path, `--skip-sklearn` to skip the scikit-learn baseline, or
     `--skip-emp-gpu` to skip the GPU empirical SD-KDE run.
   - Use `--mixture-index 0|1|2` to select which mixture from
     `kde_utils.py` you want to evaluate.
