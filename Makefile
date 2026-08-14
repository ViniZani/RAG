VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Define o cache do UV dinamicamente na pasta sgoinfre do projeto
export UV_CACHE_DIR = $(CURDIR)/.cache/uv

.PHONY: all install clean run debug lint lint-strict fclean re

all: install

install:
	# Deixa o UV gerenciar a criação da venv e a sincronização das dependências de forma limpa
	uv venv $(VENV)
	uv pip install --upgrade pip
	uv sync
	uv add --dev flake8 mypy langchain-chroma langchain-text-splitters langchain_community langchain_openai langchain_core bm25s fire

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

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
	rm -rf $(VENV) uv.lock .hf_cache .cache

re: fclean install