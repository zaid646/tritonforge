.PHONY: help sync lint test test-cpu test-gpu kernels e2e probe sanitize report gate clean

PYTHON ?= python
REMOTE ?= colab-t4
DEST ?= /content/tritonforge

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sed 's/:.*//' | sort | uniq

sync:
	REMOTE=$(REMOTE) DEST=$(DEST) ./scripts/sync.sh

lint:
	ruff check .

test: test-cpu

test-cpu:
	$(PYTHON) -m pytest -m "not gpu"

test-gpu:
	ssh $(REMOTE) "cd $(DEST) && PYTHONPATH=src $(PYTHON) -m pytest tests/ -q"

kernels:
	PYTHONPATH=src $(PYTHON) benchmarks/run_kernels.py

probe:
	PYTHONPATH=src $(PYTHON) benchmarks/probe_dequant.py

e2e:
	PYTHONPATH=src $(PYTHON) benchmarks/run_e2e.py

sanitize:
	SAN=/usr/local/cuda/bin/compute-sanitizer PYTHONPATH=src ./scripts/sanitize.sh

report:
	PYTHONPATH=src $(PYTHON) scripts/make_report.py

gate:
	$(PYTHON) scripts/check_regression.py

clean:
	rm -rf results/raw artifacts/sanitizer .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
