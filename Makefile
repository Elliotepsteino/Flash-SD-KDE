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

.PHONY: test test.all test.small test.large test.fast test.full test.unit test.integration \
	bench bench.mnist_ood bench.toy_1d_oracle \
	plot plot.mnist_ood plot.grids plot.toy_1d_oracle \
	paper paper.clean full_paper full_paper_experiments_plots toy_1d_oracle_plots oracle_16d_plots paper.figures.sync \
	plots.runtime plots.runtime.from_logs plots.runtime.util \
	run.sweep run.nd_runtime_sweep run.triton_scaling run.triton_sd_kde_nd \
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
