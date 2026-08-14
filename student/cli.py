from pathlib import Path

import fire

from .evaluation import evaluate as evaluate_fn
from .generation import answer as answer_fn
from .generation import answer_dataset as answer_dataset_fn
from .index import build_index, load_index
from .ingestion import ingest_repo
from .retrieval import search as search_fn
from .retrieval import search_dataset as search_dataset_fn


class StudentCLI:
    """CLI do projeto RAG against the machine (student side)."""

    def index(self, repo_path: str = "data/raw/vllm-0.10.1", max_chunk_size: int = 2000) -> None:
        """Indexa o repositório, construindo os índices BM25 de code e docs."""
        chunks = ingest_repo(repo_path, max_chunk_size)
        code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
        docs_chunks = [c for c in chunks if c["chunk_type"] == "docs"]

        build_index(code_chunks, Path("data/processed/bm25_index_code"))
        build_index(docs_chunks, Path("data/processed/bm25_index_docs"))

        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10, index_type: str = "docs") -> None:
        """Busca uma query única no índice indicado."""
        retriever = load_index(Path(f"data/processed/bm25_index_{index_type}"))
        results = search_fn(query, retriever, k)
        for i, chunk in enumerate(results, start=1):
            print(f"[{i}] {chunk['file_path']} ({chunk['first_character_index']}-{chunk['last_character_index']})")
            print(chunk["text"][:200])
            print("-" * 40)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_type: str = "docs",
    ) -> None:
        """Processa múltiplas perguntas de um dataset e salva os resultados de busca."""
        search_dataset_fn(Path(dataset_path), k, save_directory, index_type)

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10) -> None:
        """Avalia os resultados de busca do aluno contra o gabarito."""
        evaluate_fn(student_answer_path, dataset_path, k)

    def answer(self, query: str, k: int = 10, index_type: str = "docs") -> None:
        """Responde uma pergunta única usando o LLM com contexto recuperado."""
        retriever = load_index(Path(f"data/processed/bm25_index_{index_type}"))
        result = answer_fn(query, retriever, k)
        print(result)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Gera respostas a partir de resultados de busca já salvos."""
        answer_dataset_fn(Path(student_search_results_path), save_directory)


if __name__ == "__main__":
    fire.Fire(StudentCLI)
