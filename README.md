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
   python benchmark_triton_kde.py --seeds 0,1,2 --n-train 5000 --n-test 500 --mixture-index 0 --device cuda
   ```
   - Add `--cpu-only` if CUDA isn’t available and you just want to test the CPU
     path, `--skip-sklearn` to skip the scikit-learn baseline, or
     `--skip-emp-gpu` to skip the GPU empirical SD-KDE run.
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
