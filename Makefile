PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
ARM ?= W32
SEED ?= 0

.PHONY: help setup lint test data train g0 eval report clean

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## Create the venv and install the package with dev and reference extras
	uv venv --python 3.12 && uv pip install -e ".[dev,ref]"

lint: ## Lint and format-check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

test: ## Test suite (CPU, no network)
	$(PYTEST) tests

data: ## Download, annotate cells, build the peak matrix and the variant sets (idempotent)
	$(PY) -m ctb.data.sources
	$(PY) -m ctb.data.tenk10k
	$(PY) -m ctb.data.annotate
	$(PY) -m ctb.data.peaks
	$(PY) -m ctb.data.variants

train: ## Train one arm and seed: make train ARM=W32 SEED=0
	nice -n 15 $(PY) -m ctb.train.loop --arm $(ARM) --seed $(SEED)

g0: ## Gate G0 reference: scvi.external.SCBASSET on the same split
	nice -n 15 $(PY) -m ctb.train.scvi_reference

eval: ## Score variants and probes for every trained run, then aggregate with bootstrap intervals
	$(PY) -m ctb.eval.score_all
	$(PY) -m ctb.eval.aggregate

report: ## Regenerate every figure and table from committed results
	$(PY) -m ctb.eval.figures

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
