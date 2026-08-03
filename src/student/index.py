from pathlib import Path
import bm25s


def build_index(chunks: list[dict], save_dir: Path) -> None:
    """Builds a BM25 index from a list of chunks and persists it to disk.
    Recieve the chunks: List of chunk dicts, each with at least a "text" key.
    save_dir: Directory where the index (and corpus) will be saved."""
    texts: list[str] = []
    for c in chunks:
        texts.append(c["text"])

    corpus_tokens = bm25s.tokenize(texts)
    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(corpus_tokens)
    save_dir.mkdir(parents=True, exist_ok=True)

    retriever.save(save_dir, corpus=chunks)


def load_index(save_dir: Path) -> bm25s.BM25:
    """Loads a previously built BM25 index from disk.
        reciev a Directory where the index was saved. and
        return the loaded BM25 retriever, with corpus (chunk dicts) available.
        """
    return bm25s.BM25.load(save_dir, load_corpus=True)
