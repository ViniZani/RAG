# Sprint 2:
import bm25s
import json
from pathlib import Path
from tqdm import tqdm
from .index import load_index
from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,)


def search(query: str, retriever: bm25s.BM25, k: int) -> list[dict]:
    """Searches the given BM25 retriever for the top-k most relevant chunks.
        query: The search query text.
        retriever: A loaded BM25 retriever (with corpus attached).
        k: Number of results to return
        Than Return a list of the top-k chunk dicts, ordered by relevance."""
    try:
        tokenized_query = bm25s.tokenize(query)
        results, scores = retriever.retrieve(tokenized_query, k=k)
    except ValueError:
        if retriever.corpus is not None:
            corpus_size = len(retriever.corpus)
        else:
            corpus_size = 0
        adjusted_k = min(k, corpus_size)
        print(
            f"Aviso: k={k} bigger than corpus ({corpus_size} chunks). "
            f"Ajusted k={adjusted_k}."
        )
        if adjusted_k == 0:
            return []
        results, scores = retriever.retrieve(tokenized_query, k=adjusted_k)
    top_chunks = results[0]
    return top_chunks


def search_dataset(dataset_path: Path, k: int, save_directory: str,
                   index_type: str) -> None:
    """Runs search() for every question in a dataset and saves the results.
        dataset_path: Path to the input RagDataset JSON file.
        k: Number of results to retrieve per question.
        save_directory: Directory where the output JSON will be saved.
        index_type: Which index to use, "code" or "docs"."""
    # 1. Carrega o índice certo, com base em index_type
    #    (dica: você já tem load_index(save_dir: Path) -> bm25s.BM25 pronto;
    #    o caminho segue o padrão "data/processed/bm25_index_{index_type}")
    retriever = load_index(Path(f"data/processed/bm25_index_{index_type}"))

    # 2. Carrega e valida o dataset de entrada contra RagDataset
    #    (dica: leia o JSON com json.load, depois RagDataset(**dados)
    #    -- ou explore RagDataset.model_validate_json diretamente no arquivo,
    #    veja qual dos dois te parece mais direto)
    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)
        dataset: RagDataset = RagDataset(**raw_data)

    # 3. Roda search() para cada pergunta, com barra de progresso
    minimal_results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Buscando"):
        chunks = search(question.question, retriever, k)

        # 4. Converte cada chunk (dict) em um MinimalSource
        sources = []
        for chunk in chunks:
            sources += [
             MinimalSource(
                          file_path=chunk['file_path'],
                          first_character_index=chunk['first_character_index'],
                          last_character_index=chunk['last_character_index'],)]

        minimal_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )

    # 5. Monta o resultado final
    output = StudentSearchResults(search_results=minimal_results, k=k)

    # 6. Salva em save_directory, com o MESMO nome do arquivo de dataset_path
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / Path(dataset_path).name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"Saved student_search_results to {output_path}")
