from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def persist_documents_to_vectorstore(documents, persist_directory="./chroma_db"):
    """Gera embeddings e salva os documentos em um banco vetorial local."""

    # 1. Escolhe o modelo de embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Cria o banco vetorial a partir dos seus chunks e salva no disco
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"✅ {len(documents)} chunks salvos com sucesso no ChromaDB!")
    return vector_store
