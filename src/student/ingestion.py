from pathlib import Path
from langchain_text_splitters import (MarkdownTextSplitter,
                                      PythonCodeTextSplitter)
from langchain_core.documents import Document
import re


def chunk_python_code(text: str, max_chunk_size: int) -> list[Document]:
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if max_chunk_size > 2000:
        raise ValueError("max_chunk_size must be a lower than 2000")
    splitter = PythonCodeTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    raw_documents = splitter.create_documents([text])
    final_documents = []
    for doc in raw_documents:
        final_documents.append(
            Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata)
            )
        )
    return final_documents


def chunk_text(text: str, max_chunk_size: int = 2000) -> list[Document]:
    """Splits a Markdown file by sections, protecting code blocks, and
    indexes each chunk with its correct start_index in the original text."""

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if max_chunk_size > 2000:
        raise ValueError("max_chunk_size must be a lower than 2000")
    # 1. Proteger os blocos de código no texto inteiro com placeholders
    code_blocks = []

    def replacer(match):
        idx = len(code_blocks)
        original_text = match.group(0)
        start = match.start()
        end = match.end()

        placeholder = f"\x00CODE{idx}\x00"
        code_blocks.append({
            "start": start,
            "end": end,
            "original": original_text,
            "placeholder": placeholder,
            "len_diff": len(original_text) - len(placeholder)
        })
        return placeholder

    processed_text = re.sub(r"```.*?```", replacer, text, flags=re.DOTALL)

    # 2. Encontrar os índices de divisão por seção (headers #) no texto processado
    # Como os blocos viraram placeholders, não precisamos mais nos preocupar com '#' dentro de código!
    header_pattern = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
    split_indices = [m.start() for m in header_pattern.finditer(processed_text)]

    # Garantir que o início e o fim do texto entrem nos limites das seções
    if not split_indices or split_indices[0] != 0:
        split_indices.insert(0, 0)
    split_indices.append(len(processed_text))

    splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )

    final_documents = []

    # 3. Processar seção por seção de forma isolada
    for i in range(len(split_indices) - 1):
        section_start = split_indices[i]
        section_end = split_indices[i+1]
        section_text = processed_text[section_start:section_end]

        if not section_text.strip():
            continue

        # Rodar o splitter na seção isolada
        raw_documents = splitter.create_documents([section_text])

        for doc in raw_documents:
            content = doc.page_content
            # O start_index retornado pelo splitter é relativo à *seção*
            splitter_start_index_in_sec = doc.metadata.get("start_index", 0)

# Posição absoluta do início do chunk no texto processado (com placeholders)
            absol_splitter_index = section_start + splitter_start_index_in_sec

# 4. Calcular o delta acumulado considerando os blocos anteriores a esta posição absoluta
            delta_acumulado = 0
            for block in code_blocks:
                if block["start"] < (absol_splitter_index + delta_acumulado):
                    delta_acumulado += block["len_diff"]

            # O start_index real no texto original completo
            real_start_index = absol_splitter_index + delta_acumulado

# 5. Restaurar os placeholders de volta para os blocos de código originais
            for block in code_blocks:
                if block["placeholder"] in content:
                    content = content.replace(block["placeholder"], block["original"])

            new_metadata = dict(doc.metadata)
            new_metadata["start_index"] = real_start_index

            final_documents.append(Document(page_content=content, metadata=new_metadata))

    return final_documents


def ingest_repo(repo_path: Path, max_chunk_size: int) -> list[dict]:
    """Screapes the llm's repo, and search by .py and .md files
    parser each type of file and returns in a organized data"""
    all_chunks: list[dict] = []

    for py_file in repo_path.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for doc in chunk_python_code(content, max_chunk_size):
            all_chunks.append({
                "text": doc.page_content,
                "file_path": str(py_file),
                "first_character_index": doc.metadata["start_index"],
                "last_character_index": (doc.metadata["start_index"]
                                         + len(doc.page_content)),
                "chunk_type": "code",
            })

    for md_file in repo_path.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for doc in chunk_text(content, max_chunk_size):
            all_chunks.append({
                "text": doc.page_content,
                "file_path": str(md_file),
                "first_character_index": doc.metadata["start_index"],
                "last_character_index": (doc.metadata["start_index"]
                                         + len(doc.page_content)),
                "chunk_type": "docs",
            })
    return all_chunks


if __name__ == "__main__":
    print(ingest_repo(Path("data_test"), 200))
