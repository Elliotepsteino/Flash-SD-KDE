PY ?= python
LEGACY_DIR ?= legacy
LOG_DIR ?= file_storage/legacy_sweeps
LEGACY_FIG_DIR ?= paper/figures/legacy

.PHONY: test test.all test.small test.large test.fast test.full test.unit test.integration \
	bench bench.mnist_ood bench.toy_1d_oracle \
	plot plot.mnist_ood plot.grids plot.toy_1d_oracle \
	paper paper.clean \
	plots.legacy plots.legacy.from_logs plots.legacy.util \
	run.sweep run.nd_runtime_sweep run.triton_scaling run.triton_sd_kde_nd

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

plot: plot.mnist_ood plot.grids plot.toy_1d_oracle

plot.mnist_ood:
	$(PY) plots/plot_mnist_fashion_ood.py

plot.grids:
	$(PY) plots/save_density_ranked_grids.py

plot.toy_1d_oracle:
	$(PY) plots/plot_toy_1d_mog_oracle.py

run.sweep:
	mkdir -p $(LOG_DIR) $(LEGACY_FIG_DIR)
	bash $(LEGACY_DIR)/run_sweep.sh $(LOG_DIR)/sweep.log
	$(PY) $(LEGACY_DIR)/plot_flash_sd_kde.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/runtime_1d_kde_sdkde.pdf
	$(PY) $(LEGACY_DIR)/plot_emp_sd_kde_util.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/util_1d_empirical_sdkde.pdf

run.nd_runtime_sweep:
	mkdir -p $(LOG_DIR) $(LEGACY_FIG_DIR)
	bash $(LEGACY_DIR)/run_nd_runtime_sweep.sh $(LOG_DIR)/nd_runtime.log
	$(PY) $(LEGACY_DIR)/plot_nd_runtime.py --log $(LOG_DIR)/nd_runtime.log --output $(LEGACY_FIG_DIR)/runtime_16d_kde_sdkde.pdf

run.triton_scaling:
	mkdir -p $(LOG_DIR) $(LEGACY_FIG_DIR)
	bash $(LEGACY_DIR)/run_triton_scaling.sh $(LOG_DIR)/triton_scaling.log
	$(PY) $(LEGACY_DIR)/plot_triton_large_util.py --log $(LOG_DIR)/triton_scaling.log --output $(LEGACY_FIG_DIR)/util_1d_triton_scaling.pdf

run.triton_sd_kde_nd:
	mkdir -p $(LOG_DIR) $(LEGACY_FIG_DIR)
	bash $(LEGACY_DIR)/run_triton_sd_kde_nd.sh $(LOG_DIR)/triton_sd_kde_nd.log
	$(PY) $(LEGACY_DIR)/plot_triton_sd_kde_nd_util.py --log $(LOG_DIR)/triton_sd_kde_nd.log --output $(LEGACY_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

plots.legacy:
	mkdir -p $(LEGACY_FIG_DIR)
	@if [ -f $(LOG_DIR)/sweep.log ]; then \
	  $(PY) $(LEGACY_DIR)/plot_flash_sd_kde.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/runtime_1d_kde_sdkde.pdf; \
	  $(PY) $(LEGACY_DIR)/plot_emp_sd_kde_util.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/util_1d_empirical_sdkde.pdf; \
	else \
	  echo "Missing $(LOG_DIR)/sweep.log; skipping 1D runtime/util plots."; \
	fi
	@if [ -f $(LOG_DIR)/nd_runtime.log ]; then \
	  $(PY) $(LEGACY_DIR)/plot_nd_runtime.py --log $(LOG_DIR)/nd_runtime.log --output $(LEGACY_FIG_DIR)/runtime_16d_kde_sdkde.pdf; \
	else \
	  echo "Missing $(LOG_DIR)/nd_runtime.log; skipping 16D runtime plot."; \
	fi
	@if [ -f $(LOG_DIR)/triton_scaling.log ]; then \
	  $(PY) $(LEGACY_DIR)/plot_triton_large_util.py --log $(LOG_DIR)/triton_scaling.log --output $(LEGACY_FIG_DIR)/util_1d_triton_scaling.pdf; \
	else \
	  echo "Missing $(LOG_DIR)/triton_scaling.log; skipping Triton scaling plot."; \
	fi
	@if [ -f $(LOG_DIR)/triton_sd_kde_nd.log ]; then \
	  $(PY) $(LEGACY_DIR)/plot_triton_sd_kde_nd_util.py --log $(LOG_DIR)/triton_sd_kde_nd.log --output $(LEGACY_FIG_DIR)/util_16d_sdkde_tensorcore.pdf; \
	else \
		echo "Missing $(LOG_DIR)/triton_sd_kde_nd.log; skipping 16D utilization plot."; \
	fi

plots.legacy.from_logs:
	mkdir -p $(LEGACY_FIG_DIR)
	$(PY) $(LEGACY_DIR)/plot_flash_sd_kde.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/runtime_1d_kde_sdkde.pdf
	$(PY) $(LEGACY_DIR)/plot_emp_sd_kde_util.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/util_1d_empirical_sdkde.pdf
	$(PY) $(LEGACY_DIR)/plot_nd_runtime.py --log $(LOG_DIR)/nd_runtime.log --output $(LEGACY_FIG_DIR)/runtime_16d_kde_sdkde.pdf
	$(PY) $(LEGACY_DIR)/plot_triton_large_util.py --log $(LOG_DIR)/triton_scaling.log --output $(LEGACY_FIG_DIR)/util_1d_triton_scaling.pdf
	$(PY) $(LEGACY_DIR)/plot_triton_sd_kde_nd_util.py --log $(LOG_DIR)/triton_sd_kde_nd.log --output $(LEGACY_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

plots.legacy.util:
	mkdir -p $(LEGACY_FIG_DIR)
	$(PY) $(LEGACY_DIR)/plot_emp_sd_kde_util.py --log $(LOG_DIR)/sweep.log --output $(LEGACY_FIG_DIR)/util_1d_empirical_sdkde.pdf
	$(PY) $(LEGACY_DIR)/plot_triton_large_util.py --log $(LOG_DIR)/triton_scaling.log --output $(LEGACY_FIG_DIR)/util_1d_triton_scaling.pdf
	$(PY) $(LEGACY_DIR)/plot_triton_sd_kde_nd_util.py --log $(LOG_DIR)/triton_sd_kde_nd.log --output $(LEGACY_FIG_DIR)/util_16d_sdkde_tensorcore.pdf

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
