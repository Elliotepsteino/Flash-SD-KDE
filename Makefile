PY ?= python

.PHONY: test test.all test.small test.large test.fast test.full test.unit test.integration bench bench.mnist_ood bench.toy_1d_oracle plot plot.mnist_ood plot.grids plot.toy_1d_oracle paper paper.clean

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
