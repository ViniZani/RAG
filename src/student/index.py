from pathlib import Path
import bm25s


def build_index(chunks: list[dict], save_dir: Path) -> None:
    """Builds a BM25 index from a list of chunks and persists it to disk.
    Recieve the chunks: List of chunk dicts, each with at least a "text" key.
    save_dir: Directory where the index (and corpus) will be saved."""
    # 1. Extrai só os textos dos chunks, pra tokenizar
    #    (lembra: bm25s.tokenize espera uma lista de strings)
    texts: list[str] = []
    for c in chunks:
        texts.append(c["text"])

    # 2. Tokeniza o corpus inteiro
    corpus_tokens = bm25s.tokenize(texts)

    # 3. Cria o retriever, passando os CHUNKS (dicts inteiros) como corpus
    #    -> isso é o que garante que retrieve() devolva seus dicts direto,
    #       não IDs numéricos soltos
    retriever = bm25s.BM25(corpus=chunks)

    # 4. Indexa
    retriever.index(corpus_tokens)

    # 5. Garante que a pasta existe antes de salvar(parents=True,exist_ok=True)
    ...
    save_dir.mkdir(parents=True, exist_ok=True)

    # 6. Salva o índice E o corpus (sem `corpus=`, você perde os dicts no load)
    retriever.save(save_dir, corpus=chunks)


def load_index(save_dir: Path) -> bm25s.BM25:
    """Loads a previously built BM25 index from disk.
        reciev a Directory where the index was saved. and
        return the loaded BM25 retriever, with corpus (chunk dicts) available.
        """
    return bm25s.BM25.load(save_dir, load_corpus=True)
