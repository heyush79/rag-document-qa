.PHONY: install run test lint ingest clean

# Needs Python 3.11 to 3.13, e.g. `make install PYTHON=python3.12`.
PYTHON ?= python3

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -e ".[local,dev]"

run:
	. .venv/bin/activate && uvicorn app.main:create_app --factory --reload

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check app tests

ingest:
	. .venv/bin/activate && python scripts/ingest_sample.py

clean:
	rm -rf data/rag.db data/rag.db-* data/chroma .pytest_cache .ruff_cache .coverage htmlcov
