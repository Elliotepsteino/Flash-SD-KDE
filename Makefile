PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
RUNTIME_DIR ?= experiments/runtime
RUNTIME_CFG_DIR ?= experiments/configs/runtime
RUNTIME_LOG_DIR ?= file_storage/runtime_sweeps
RUNTIME_FIG_DIR ?= paper/figures/runtime
FILE_STORAGE_ROOT ?= file_storage
PAPER_PLOTS_DIR ?= $(FILE_STORAGE_ROOT)/paper_plots
PAPER_PLOTS_RUN ?= $(PAPER_PLOTS_DIR)/$(shell date +%Y%m%d_%H%M%S)
PAPER_PLOTS_BASELINE ?= $(PAPER_PLOTS_RUN)/baseline
PAPER_PLOTS_OUT ?= $(PAPER_PLOTS_RUN)/generated
TOY_1D_PLOTS_OUT ?= $(PAPER_PLOTS_OUT)
REBUTTAL_FIG1_START_POWER ?= 11
REBUTTAL_FIG1_END_POWER ?= 15
REBUTTAL_FIG1_SEED ?= 0
REBUTTAL_FIG1_WARMUP ?= 1
REBUTTAL_FIG1_FLASH_REPEATS ?= 10
REBUTTAL_FIG1_BASELINE_REPEATS ?= 3
REBUTTAL_QUERY_SWEEP_N_TRAIN ?= 32768
REBUTTAL_QUERY_SWEEP_N_TEST_LIST ?= 4,16,64,256,1024,4096,16384
REBUTTAL_QUERY_SWEEP_WARMUP ?= 1
REBUTTAL_QUERY_SWEEP_FLASH_REPEATS ?= 10
REBUTTAL_QUERY_SWEEP_BASELINE_REPEATS ?= 3
REBUTTAL_QUERY_SWEEP_NONFUSED_CHUNK_SIZE ?= 2048
REBUTTAL_ICML2026_FLASH_LAPLACE_OUT_DIR ?= $(FILE_STORAGE_ROOT)/error_suite_16d/rebuttal_icml2026_flash_laplace_$(shell date +%Y%m%d_%H%M%S)
REBUTTAL_OPERATOR_OUT_DIR ?= $(PAPER_PLOTS_OUT)
REBUTTAL_EMBED_OUT_DIR ?= $(FILE_STORAGE_ROOT)/benchmarks/mnist_fashion_pca64_similarity
REBUTTAL_REPORT_RUNTIME_JSON ?= $(FILE_STORAGE_ROOT)/paper_plots/rebuttal_full_20260330/generated/fig_rebuttal_runtime_16d_kde_sdkde.json
REBUTTAL_REPORT_NEG_CSV ?= $(FILE_STORAGE_ROOT)/error_suite_16d/rebuttal_icml2026_full/results.csv
REBUTTAL_C4_EMBED_DEVICE ?= cuda
REBUTTAL_C4_EMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
REBUTTAL_C4_EMBED_DATASET ?= allenai/c4
REBUTTAL_C4_EMBED_CONFIG ?= en
REBUTTAL_C4_EMBED_SPLIT ?= train
REBUTTAL_C4_EMBED_SAMPLES ?= 100
REBUTTAL_C4_EMBED_MIN_TOKENS ?= 128
REBUTTAL_C4_EMBED_MAX_TOKENS ?= 256
REBUTTAL_C4_EMBED_BATCH_SIZE ?= 32
REAL_APP_TULU_EMBED_DEVICE ?= cuda
REAL_APP_TULU_EMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
REAL_APP_TULU_EMBED_DATASET ?= allenai/tulu-3-sft-mixture
REAL_APP_TULU_EMBED_SPLIT ?= train
REAL_APP_TULU_EMBED_SUBSET ?= train_pool
REAL_APP_TULU_EMBED_VALIDATION_SIZE ?= 50000
REAL_APP_TULU_EMBED_SEED ?= 20260330
REAL_APP_TULU_EMBED_SAMPLES ?= 100
REAL_APP_TULU_EMBED_MAX_TOKENS ?= 256
REAL_APP_TULU_EMBED_BATCH_SIZE ?= 256

.PHONY: test test.all test.small test.large test.fast test.full test.unit test.integration \
	bench bench.mnist_ood bench.toy_1d_oracle bench.pykeops_16d \
	plot plot.mnist_ood plot.grids plot.toy_1d_oracle \
	paper paper.clean full_paper full_paper_experiments_plots toy_1d_oracle_plots oracle_16d_plots paper.figures.sync \
	plots.runtime plots.runtime.from_logs plots.runtime.util \
	run.sweep run.nd_runtime_sweep run.triton_scaling run.triton_sd_kde_nd \
	rebuttal.figure1_16d_runtime rebuttal.icml2026.query_batching_sweep rebuttal.icml2026.flash_laplace_negative_mass \
	rebuttal.icml2026.operator_ablation rebuttal.icml2026.embedding_similarity \
	rebuttal.icml2026.overall_report rebuttal.icml2026.c4_embedding_sanity \
	real_application.tulu_sft_embeddings \
	bench.emp_score_kernel_speed

test: test.small test.large
test.all: test.small test.large
test.fast: test.small
test.full: test.large
test.unit: test.small
test.integration: test.large

test.small:
	$(PY) -m pytest -m small

test.large:
	$(PY) -m pytest -m large

bench: bench.mnist_ood bench.toy_1d_oracle

bench.mnist_ood:
	$(PY) benchmarks/mnist_fashion_pca16_ood.py

bench.toy_1d_oracle:
	$(PY) benchmarks/toy_1d_mog_oracle.py

bench.pykeops_16d:
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) -m experiments.runtime.benchmark_pykeops_16d \
		--n-train 32768 \
		--n-test 4096 \
		--device cuda \
		--seed 0 \
		--flash-repeats 10 \
		--output "$(PAPER_PLOTS_OUT)/pykeops_16d_runtime.json" \
		--table-output "$(PAPER_PLOTS_OUT)/table_pykeops_16d_runtime.txt"

bench.emp_score_kernel_speed:
	$(PY) -m experiments.runtime.compare_emp_score_kernels --n-train 32768

plot: plot.mnist_ood plot.grids plot.toy_1d_oracle

plot.mnist_ood:
	$(PY) plots/plot_mnist_fashion_ood.py

plot.grids:
	$(PY) plots/save_density_ranked_grids.py

plot.toy_1d_oracle:
	$(PY) plots/plot_toy_1d_mog_oracle.py

run.sweep:
	mkdir -p $(RUNTIME_LOG_DIR) $(RUNTIME_FIG_DIR)
	bash $(RUNTIME_CFG_DIR)/run_sweep.sh $(RUNTIME_LOG_DIR)/sweep.log
	$(PY) $(RUNTIME_DIR)/plot_flash_sd_kde.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/runtime_1d_kde_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_emp_sd_kde_util.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/util_1d_empirical_sdkde.pdf

run.nd_runtime_sweep:
	mkdir -p $(RUNTIME_LOG_DIR) $(RUNTIME_FIG_DIR)
	bash $(RUNTIME_CFG_DIR)/run_nd_runtime_sweep.sh $(RUNTIME_LOG_DIR)/nd_runtime.log
	$(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.png

run.triton_scaling:
	mkdir -p $(RUNTIME_LOG_DIR) $(RUNTIME_FIG_DIR)
	bash $(RUNTIME_CFG_DIR)/run_triton_scaling.sh $(RUNTIME_LOG_DIR)/triton_scaling.log
	$(PY) $(RUNTIME_DIR)/plot_triton_large_util.py --log $(RUNTIME_LOG_DIR)/triton_scaling.log --output $(RUNTIME_FIG_DIR)/util_1d_triton_scaling.pdf

run.triton_sd_kde_nd:
	mkdir -p $(RUNTIME_LOG_DIR) $(RUNTIME_FIG_DIR)
	bash $(RUNTIME_CFG_DIR)/run_triton_sd_kde_nd.sh $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log
	$(PY) $(RUNTIME_DIR)/plot_triton_sd_kde_nd_util.py --log $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log --output $(RUNTIME_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

plots.runtime:
	mkdir -p $(RUNTIME_FIG_DIR)
	@if [ -f $(RUNTIME_LOG_DIR)/sweep.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_flash_sd_kde.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/runtime_1d_kde_sdkde.pdf; \
	  $(PY) $(RUNTIME_DIR)/plot_emp_sd_kde_util.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/util_1d_empirical_sdkde.pdf; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/sweep.log; skipping 1D runtime/util plots."; \
	fi
	@if [ -f $(RUNTIME_LOG_DIR)/nd_runtime.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.pdf; \
	  $(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.png; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/nd_runtime.log; skipping 16D runtime plot."; \
	fi
	@if [ -f $(RUNTIME_LOG_DIR)/triton_scaling.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_triton_large_util.py --log $(RUNTIME_LOG_DIR)/triton_scaling.log --output $(RUNTIME_FIG_DIR)/util_1d_triton_scaling.pdf; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/triton_scaling.log; skipping Triton scaling plot."; \
	fi
	@if [ -f $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_triton_sd_kde_nd_util.py --log $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log --output $(RUNTIME_FIG_DIR)/util_16d_sdkde_tensorcore.pdf; \
	else \
		echo "Missing $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log; skipping 16D utilization plot."; \
	fi

plots.runtime.from_logs:
	mkdir -p $(RUNTIME_FIG_DIR)
	$(PY) $(RUNTIME_DIR)/plot_flash_sd_kde.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/runtime_1d_kde_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_emp_sd_kde_util.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/util_1d_empirical_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(RUNTIME_FIG_DIR)/runtime_16d_kde_sdkde.png
	$(PY) $(RUNTIME_DIR)/plot_triton_large_util.py --log $(RUNTIME_LOG_DIR)/triton_scaling.log --output $(RUNTIME_FIG_DIR)/util_1d_triton_scaling.pdf
	$(PY) $(RUNTIME_DIR)/plot_triton_sd_kde_nd_util.py --log $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log --output $(RUNTIME_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

plots.runtime.util:
	mkdir -p $(RUNTIME_FIG_DIR)
	$(PY) $(RUNTIME_DIR)/plot_emp_sd_kde_util.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(RUNTIME_FIG_DIR)/util_1d_empirical_sdkde.pdf
	$(PY) $(RUNTIME_DIR)/plot_triton_large_util.py --log $(RUNTIME_LOG_DIR)/triton_scaling.log --output $(RUNTIME_FIG_DIR)/util_1d_triton_scaling.pdf
	$(PY) $(RUNTIME_DIR)/plot_triton_sd_kde_nd_util.py --log $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log --output $(RUNTIME_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

PAPER_DIR := paper
PAPER_BUILD := $(PAPER_DIR)/build
PAPER_MAIN := $(PAPER_DIR)/main.tex

paper:
	mkdir -p $(PAPER_BUILD)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
	cd $(PAPER_DIR) && bibtex build/main
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex

paper.clean:
	rm -rf $(PAPER_BUILD)

full_paper:
	mkdir -p $(PAPER_PLOTS_BASELINE) $(PAPER_PLOTS_OUT)
	cp -a $(PAPER_DIR)/figures/. $(PAPER_PLOTS_BASELINE)/
	@if [ -f $(RUNTIME_LOG_DIR)/sweep.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_flash_sd_kde.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(PAPER_PLOTS_OUT)/runtime_1d_kde_sdkde.pdf; \
	  $(PY) $(RUNTIME_DIR)/plot_emp_sd_kde_util.py --log $(RUNTIME_LOG_DIR)/sweep.log --output $(PAPER_PLOTS_OUT)/util_1d_empirical_sdkde.pdf; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/sweep.log; skipping 1D runtime/util plots."; \
	fi
	@if [ -f $(RUNTIME_LOG_DIR)/nd_runtime.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(PAPER_PLOTS_OUT)/runtime_16d_kde_sdkde.pdf; \
	  $(PY) $(RUNTIME_DIR)/plot_nd_runtime.py --log $(RUNTIME_LOG_DIR)/nd_runtime.log --output $(PAPER_PLOTS_OUT)/runtime_16d_kde_sdkde.png; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/nd_runtime.log; skipping 16D runtime plot."; \
	fi
	@if [ -f $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log ]; then \
	  $(PY) $(RUNTIME_DIR)/plot_triton_sd_kde_nd_util.py --log $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log --output $(PAPER_PLOTS_OUT)/util_16d_sdkde_tensorcore.pdf; \
	else \
	  echo "Missing $(RUNTIME_LOG_DIR)/triton_sd_kde_nd.log; skipping 16D utilization plot."; \
	fi
	@$(PY) scripts/full_paper_collect.py --output "$(PAPER_PLOTS_OUT)"

full_paper_experiments_plots:
	@echo "Running experiments needed for paper plots (may take significant time/GPU)."
	$(MAKE) run.sweep run.nd_runtime_sweep run.triton_sd_kde_nd RUNTIME_FIG_DIR=$(PAPER_PLOTS_OUT) PAPER_PLOTS_RUN=$(PAPER_PLOTS_RUN)
	$(MAKE) bench.toy_1d_oracle PAPER_PLOTS_RUN=$(PAPER_PLOTS_RUN)
	$(MAKE) bench.pykeops_16d PAPER_PLOTS_RUN=$(PAPER_PLOTS_RUN)
	$(PY) -m experiments.error_suite_16d.sweep --config configs/error_suite_16d/grid_oracle_mog_16d.yaml
	$(MAKE) full_paper PAPER_PLOTS_RUN=$(PAPER_PLOTS_RUN)

toy_1d_oracle_plots:
	@echo "Running toy 1D oracle benchmark + plots."
	mkdir -p $(TOY_1D_PLOTS_OUT)
	$(MAKE) bench.toy_1d_oracle PAPER_PLOTS_RUN=$(PAPER_PLOTS_RUN)
	$(PY) scripts/toy_1d_oracle_paper_plots.py --output "$(TOY_1D_PLOTS_OUT)"
	@echo "Toy 1D oracle plots written to $(TOY_1D_PLOTS_OUT)"

oracle_16d_plots:
	@echo "Running 16D oracle sweep + plots."
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) -m experiments.error_suite_16d.sweep --config configs/error_suite_16d/grid_oracle_mog_16d.yaml
	$(PY) scripts/error_suite_oracle_plot.py --output "$(PAPER_PLOTS_OUT)"
	@echo "16D oracle plots written to $(PAPER_PLOTS_OUT)"

paper.figures.sync:
	@$(PY) scripts/paper_figures_sync.py --source "$(PAPER_PLOTS_OUT)" --dest "$(PAPER_DIR)/figures"

rebuttal.figure1_16d_runtime:
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) -m experiments.runtime.benchmark_rebuttal_16d_runtime \
		--start-power $(REBUTTAL_FIG1_START_POWER) \
		--end-power $(REBUTTAL_FIG1_END_POWER) \
		--seed $(REBUTTAL_FIG1_SEED) \
		--warmup $(REBUTTAL_FIG1_WARMUP) \
		--flash-repeats $(REBUTTAL_FIG1_FLASH_REPEATS) \
		--baseline-repeats $(REBUTTAL_FIG1_BASELINE_REPEATS) \
		--device cuda \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.json" \
		--markdown-output "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.md"
	$(PY) $(RUNTIME_DIR)/plot_rebuttal_16d_runtime.py \
		--input "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.json" \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.pdf"
	$(PY) $(RUNTIME_DIR)/plot_rebuttal_16d_runtime.py \
		--input "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.json" \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_runtime_16d_kde_sdkde.png"

rebuttal.icml2026.query_batching_sweep:
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) -m experiments.runtime.benchmark_rebuttal_16d_query_batching \
		--n-train $(REBUTTAL_QUERY_SWEEP_N_TRAIN) \
		--n-test-list "$(REBUTTAL_QUERY_SWEEP_N_TEST_LIST)" \
		--seed $(REBUTTAL_FIG1_SEED) \
		--warmup $(REBUTTAL_QUERY_SWEEP_WARMUP) \
		--flash-repeats $(REBUTTAL_QUERY_SWEEP_FLASH_REPEATS) \
		--baseline-repeats $(REBUTTAL_QUERY_SWEEP_BASELINE_REPEATS) \
		--nonfused-chunk-size $(REBUTTAL_QUERY_SWEEP_NONFUSED_CHUNK_SIZE) \
		--device cuda \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_query_batching_16d.json" \
		--markdown-output "$(PAPER_PLOTS_OUT)/table_rebuttal_query_batching_16d.md"
	$(PY) $(RUNTIME_DIR)/plot_rebuttal_16d_query_batching.py \
		--input "$(PAPER_PLOTS_OUT)/fig_rebuttal_query_batching_16d.json" \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_query_batching_16d.pdf"
	$(PY) $(RUNTIME_DIR)/plot_rebuttal_16d_query_batching.py \
		--input "$(PAPER_PLOTS_OUT)/fig_rebuttal_query_batching_16d.json" \
		--output "$(PAPER_PLOTS_OUT)/fig_rebuttal_query_batching_16d.png"

rebuttal.icml2026.flash_laplace_negative_mass:
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) -m experiments.error_suite_16d.sweep \
		--config configs/error_suite_16d/rebuttal_flash_laplace_negative_mass.yaml \
		--out_dir "$(REBUTTAL_ICML2026_FLASH_LAPLACE_OUT_DIR)"
	$(PY) scripts/error_suite_negative_mass_table.py \
		--results_dir "$(REBUTTAL_ICML2026_FLASH_LAPLACE_OUT_DIR)" \
		--method flash_laplace \
		--output "$(PAPER_PLOTS_OUT)/table_rebuttal_flash_laplace_negative_mass.md"

rebuttal.icml2026.operator_ablation:
	mkdir -p $(REBUTTAL_OPERATOR_OUT_DIR)
	$(PY) -m experiments.runtime.benchmark_operator_ablation \
		--device cuda \
		--start-power 11 \
		--end-power 13 \
		--warmup 1 \
		--repeats 3 \
		--output "$(REBUTTAL_OPERATOR_OUT_DIR)/rebuttal_operator_ablation.json" \
		--markdown-output "$(REBUTTAL_OPERATOR_OUT_DIR)/rebuttal_operator_ablation.md"

rebuttal.icml2026.embedding_similarity:
	$(PY) benchmarks/mnist_fashion_pca64_similarity.py \
		--device cuda \
		--pca-components 64 \
		--n-train-list 1000,2000,4000 \
		--n-id-eval 4000 \
		--n-ood-eval 4000 \
		--output-tag benchmarks/mnist_fashion_pca64_similarity

rebuttal.icml2026.c4_embedding_sanity:
	$(PY) benchmarks/c4_minilm_embedding_sanity.py \
		--device "$(REBUTTAL_C4_EMBED_DEVICE)" \
		--dataset-name "$(REBUTTAL_C4_EMBED_DATASET)" \
		--dataset-config "$(REBUTTAL_C4_EMBED_CONFIG)" \
		--split "$(REBUTTAL_C4_EMBED_SPLIT)" \
		--embedding-model "$(REBUTTAL_C4_EMBED_MODEL)" \
		--n-samples "$(REBUTTAL_C4_EMBED_SAMPLES)" \
		--min-tokens "$(REBUTTAL_C4_EMBED_MIN_TOKENS)" \
		--max-tokens "$(REBUTTAL_C4_EMBED_MAX_TOKENS)" \
		--batch-size "$(REBUTTAL_C4_EMBED_BATCH_SIZE)" \
		--output-tag benchmarks/c4_minilm_embedding_sanity

real_application.tulu_sft_embeddings:
	$(PY) benchmarks/tulu_sft_minilm_embeddings.py \
		--device "$(REAL_APP_TULU_EMBED_DEVICE)" \
		--dataset-name "$(REAL_APP_TULU_EMBED_DATASET)" \
		--split "$(REAL_APP_TULU_EMBED_SPLIT)" \
		--subset "$(REAL_APP_TULU_EMBED_SUBSET)" \
		--validation-size "$(REAL_APP_TULU_EMBED_VALIDATION_SIZE)" \
		--shuffle-seed "$(REAL_APP_TULU_EMBED_SEED)" \
		--embedding-model "$(REAL_APP_TULU_EMBED_MODEL)" \
		--n-samples "$(REAL_APP_TULU_EMBED_SAMPLES)" \
		--max-tokens "$(REAL_APP_TULU_EMBED_MAX_TOKENS)" \
		--batch-size "$(REAL_APP_TULU_EMBED_BATCH_SIZE)" \
		--output-tag benchmarks/tulu_sft_minilm_embeddings

rebuttal.icml2026.overall_report:
	mkdir -p $(PAPER_PLOTS_OUT)
	$(PY) scripts/rebuttal_icml2026_overall_report.py \
		--runtime-json "$(REBUTTAL_REPORT_RUNTIME_JSON)" \
		--negative-mass-csv "$(REBUTTAL_REPORT_NEG_CSV)" \
		--embedding-json "$(FILE_STORAGE_ROOT)/benchmarks/mnist_fashion_pca64_similarity/$(shell ls -1t $(FILE_STORAGE_ROOT)/benchmarks/mnist_fashion_pca64_similarity 2>/dev/null | head -n 1)/results.json" \
		--output "$(PAPER_PLOTS_OUT)/rebuttal_icml2026_overall_report.md"
