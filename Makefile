VENV = .venv
PYTHON = $(VENV)/bin/python

export UV_CACHE_DIR = $(CURDIR)/.cache/uv
export HF_HOME = $(CURDIR)/.cache/huggingface

.PHONY: all install run demo debug lint lint-strict clean fclean re

all: install

install:
	uv venv $(VENV)
	uv sync


run:
	uv run $(PYTHON) -m src $(if $(ARGS),$(ARGS),index --max_chunk_size 2000)


demo:
	uv run $(PYTHON) -m src search_dataset \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results/AnsweredQuestions \
		--index_type docs
	@if [ -x ./moulinette ]; then \
		./moulinette evaluate_student_search_results \
			data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
			data/datasets/AnsweredQuestions/dataset_docs_public.json \
			--k 10 --max_context_length 2000; \
	else \
		echo "[demo] moulinette não encontrada localmente -- pulando avaliação (docs)."; \
	fi
	uv run $(PYTHON) -m src answer_dataset \
		--student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer/AnsweredQuestions

	uv run $(PYTHON) -m src search_dataset \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results/AnsweredQuestions \
		--index_type code
	@if [ -x ./moulinette ]; then \
		./moulinette evaluate_student_search_results \
			data/output/search_results/AnsweredQuestions/dataset_code_public.json \
			data/datasets/AnsweredQuestions/dataset_code_public.json \
			--k 10 --max_context_length 2000; \
	else \
		echo "[demo] moulinette não encontrada localmente -- pulando avaliação (code)."; \
	fi
	uv run $(PYTHON) -m src answer_dataset \
		--student_search_results_path data/output/search_results/AnsweredQuestions/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer/AnsweredQuestions

debug:
	uv run $(PYTHON) -m pdb -m src $(ARGS)

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
	rm -rf $(VENV) uv.lock .cache
	rm -rf data/processed/*
	rm -rf data/output/*

re: fclean install