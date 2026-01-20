PY ?= python

.PHONY: test test.all test.small test.large test.fast test.full test.unit test.integration bench bench.mnist_ood plot plot.mnist_ood plot.grids

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

bench: bench.mnist_ood

bench.mnist_ood:
	$(PY) benchmarks/mnist_fashion_pca16_ood.py

plot: plot.mnist_ood plot.grids

plot.mnist_ood:
	$(PY) plots/plot_mnist_fashion_ood.py

plot.grids:
	$(PY) plots/save_density_ranked_grids.py
