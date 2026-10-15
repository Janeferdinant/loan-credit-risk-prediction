PY ?= python

.PHONY: install lint test test-all data train evaluate sweep explorer all

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

train:
	$(PY) -m credit_risk train --jobs 8

evaluate:
	$(PY) -m credit_risk evaluate --jobs 8

sweep:
	$(PY) -m credit_risk sweep --jobs 8

explorer:
	$(PY) scripts/build_explorer.py

all: train evaluate explorer
