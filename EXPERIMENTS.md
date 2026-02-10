# Experiments and Reproduction

This document contains the commands to reproduce the plots used in:
`Flash-SD-KDE: Accelerating SD-KDE with Tensor Cores`.

## Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Full Reproduction (One Command)

Run all experiment sweeps and generate the full paper plot set:

```bash
make full_paper_experiments_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

Outputs:
- Generated plots: `file_storage/paper_plots/<ts>/generated`
- Baseline snapshot of current paper figures: `file_storage/paper_plots/<ts>/baseline`

## Figure Commands (Paper Order)

Replace `<ts>` with a timestamp (example: `20260210_120000`).

1. Figure 1: 16D runtime comparison
```bash
make run.nd_runtime_sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> RUNTIME_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
Outputs:
- `runtime_16d_kde_sdkde.pdf`
- `runtime_16d_kde_sdkde.png`

2. Figure 2: 16D oracle error
```bash
make oracle_16d_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
Outputs:
- `fig_oracle_error_vs_n_16d.pdf`
- `fig_oracle_error_vs_n_16d.png`

3. Figure 3: 1D oracle error
```bash
make toy_1d_oracle_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
Outputs:
- `fig_oracle_error_vs_n_1d.pdf`
- `fig_oracle_error_vs_n_1d.png`

4. Figure 4: 1D fused vs non-fused runtime
```bash
make toy_1d_oracle_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
Outputs:
- `fig_fused_vs_nonfused_runtime.pdf`
- `fig_fused_vs_nonfused_runtime.png`

5. Figure 5: 16D utilization
```bash
make run.triton_sd_kde_nd PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> RUNTIME_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
Output:
- `util_16d_sdkde_tensorcore.pdf`

6. Figure 6: 1D runtime appendix
```bash
make run.sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> RUNTIME_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
Output:
- `runtime_1d_kde_sdkde.pdf`

7. Figure 7: 1D utilization appendix
```bash
make run.sweep PAPER_PLOTS_RUN=file_storage/paper_plots/<ts> RUNTIME_FIG_DIR=file_storage/paper_plots/<ts>/generated
```
Output:
- `util_1d_empirical_sdkde.pdf`

8. Appendix oracle plot
```bash
make oracle_16d_plots PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```
Outputs:
- `fig_oracle_error_vs_n_16d.pdf`
- `fig_oracle_error_vs_n_16d.png`

## Sync Generated Plots Into `paper/figures`

```bash
make paper.figures.sync PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

## Notes

- The reproduction targets are GPU-heavy and can take substantial time.
- `run.nd_runtime_sweep` now writes both PDF and PNG for the 16D runtime plot.
