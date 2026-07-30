from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def save_docs_vectordb(documents, persist_directory="./chroma_db"):
    """Gera embeddings e salva os documentos em um banco vetorial local."""

    # 2. Cria o banco vetorial a partir dos seus chunks e salva no disco
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
        persist_directory=persist_directory
    )

    print(f"✅ {len(documents)} chunks salvos com sucesso no ChromaDB!")
    return vector_store
