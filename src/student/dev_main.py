from pathlib import Path

from src.student.ingestion import ingest_repo
from src.student.index import build_index


def main() -> None:
    """Roda o pipeline de ingestão construído até agora e imprime um resumo."""
    # troque para "data/raw/vllm-0.10.1" quando for testar em escala real
    repo_path = "data_test"
    max_chunk_size = 2000

    chunks = ingest_repo(Path(repo_path), max_chunk_size)

    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    docs_chunks = [c for c in chunks if c["chunk_type"] == "docs"]

    print(f"Total de chunks: {len(chunks)}")
    print(f"  - código: {len(code_chunks)}")
    print(f"  - docs:   {len(docs_chunks)}")

    print("\nExemplo de chunk de código:")
    if code_chunks:
        print(code_chunks[0])

    print("\nExemplo de chunk de docs:")
    if docs_chunks:
        print(docs_chunks[0])

    build_index(code_chunks, Path("data/processed/bm25_index_code"))
    build_index(docs_chunks, Path("data/processed/bm25_index_docs"))


if __name__ == "__main__":
    # Para rodar: uv run python -m src.student.dev_main
    main()
