# Sprint 2:
def search(query: str, retriever: bm25s.BM25, k: int) -> list[dict]:
    """..."""
    # 1. Tokeniza a query (mesma função bm25s.tokenize, mas numa string só)
    # 2. Chama retriever.retrieve(...)
    # 3. Extrai a "linha 0" do resultado (lembra do shape (n_queries, k)?)
    # 4. Retorna a lista de chunks (dicts) encontrados
