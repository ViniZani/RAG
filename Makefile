VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all install clean run debug lint lint-strict fclean re

all: install

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install langchain-text-splitters vllm
	$(PIP) install langchain-chroma langchain-openai
	uv sync
	uv add --dev flake8 mypy

run:
	python -m src $(ARGS)
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run flake8 --exclude $(VENV) .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 --exclude $(VENV) .
	uv run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

fclean: clean
	rm -rf $(VENV) uv.lock .hf_cache

re: fclean install
