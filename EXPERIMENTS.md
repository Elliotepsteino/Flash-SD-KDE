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

## Rebuttal ICML 2026

### Figure 1

Runtime comparison for 16-D KDE / SD-KDE across `n_train <= 32768` with
`n_test = n_train / 8`, including sklearn KDE, SD-KDE (Torch), SD-KDE
(PyKeOps), and Flash-SD-KDE:

```bash
make rebuttal.figure1_16d_runtime PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

Outputs:
- `fig_rebuttal_runtime_16d_kde_sdkde.json`
- `fig_rebuttal_runtime_16d_kde_sdkde.md`
- `fig_rebuttal_runtime_16d_kde_sdkde.pdf`
- `fig_rebuttal_runtime_16d_kde_sdkde.png`

### Flash-Laplace-KDE Negative-Mass Table

Runs a flash-laplace-only 16-D oracle sweep and aggregates the signed-density
diagnostics into a Markdown table for rebuttal text.

```bash
make rebuttal.icml2026.flash_laplace_negative_mass PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

Output:
- `table_rebuttal_flash_laplace_negative_mass.md`

### Operator Ablation

Benchmarks exact SD-KDE in eager Torch, exact SD-KDE with `torch.compile`,
Flash-SD-KDE, and Flash-Laplace-KDE, together with a pass-level operator table
covering GEMMs, exponentials, reductions, and atomics.

```bash
make rebuttal.icml2026.operator_ablation PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

Outputs:
- `rebuttal_operator_ablation.json`
- `rebuttal_operator_ablation.md`

### PCA-64 Embedding Similarity / Divergence

Runs an MNIST vs Fashion-MNIST benchmark in PCA-reduced 64-D feature space,
reporting runtime plus OOD-style quality metrics.

```bash
make rebuttal.icml2026.embedding_similarity
```

Outputs:
- `file_storage/benchmarks/mnist_fashion_pca64_similarity/<run>/results.json`
- `file_storage/benchmarks/mnist_fashion_pca64_similarity/<run>/report.md`

### C4 MiniLM Embedding Sanity Check

Runs the first stage of a larger real-world application benchmark: stream C4,
keep 100 samples with lengths in the `128..256` token range, truncate each
sample to at most 256 tokens, embed them with
`sentence-transformers/all-MiniLM-L6-v2`, and persist both the selected texts
and the resulting `384`-D embeddings to disk.

```bash
make rebuttal.icml2026.c4_embedding_sanity
```

Outputs:
- `file_storage/benchmarks/c4_minilm_embedding_sanity/<run>/embeddings.npy`
- `file_storage/benchmarks/c4_minilm_embedding_sanity/<run>/records.jsonl`
- `file_storage/benchmarks/c4_minilm_embedding_sanity/<run>/results.json`
- `file_storage/benchmarks/c4_minilm_embedding_sanity/<run>/report.md`

### Tulu-3 SFT MiniLM Embeddings

Creates deterministic MiniLM embeddings for a shuffled subset of
`allenai/tulu-3-sft-mixture`. The current local dataset has `939,343` rows, so
with a fixed `50,000`-example validation split the available train pool is
`889,343` rows. Each SFT sample is rendered as one conversation string with
role prefixes, truncated to `256` tokens, embedded in `384` dimensions, and
saved together with the original dataset indices needed for later subset
selection and fine-tuning.

```bash
make real_application.tulu_sft_embeddings
```

Outputs:
- `file_storage/benchmarks/tulu_sft_minilm_embeddings/<run>/embeddings.npy`
- `file_storage/benchmarks/tulu_sft_minilm_embeddings/<run>/dataset_indices.npy`
- `file_storage/benchmarks/tulu_sft_minilm_embeddings/<run>/records.jsonl`
- `file_storage/benchmarks/tulu_sft_minilm_embeddings/<run>/results.json`
- `file_storage/benchmarks/tulu_sft_minilm_embeddings/<run>/report.md`

### Overall Report

Aggregates the rebuttal runtime figure, Flash-Laplace negative-mass table,
operator ablation, and PCA-64 embedding benchmark into one Markdown report.

```bash
make rebuttal.icml2026.overall_report PAPER_PLOTS_RUN=file_storage/paper_plots/<ts>
```

Output:
- `rebuttal_icml2026_overall_report.md`

## Notes

- The reproduction targets are GPU-heavy and can take substantial time.
- `run.nd_runtime_sweep` now writes both PDF and PNG for the 16D runtime plot.
