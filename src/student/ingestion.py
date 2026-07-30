from pathlib import Path
from langchain_text_splitters import (MarkdownTextSplitter,
                                      PythonCodeTextSplitter)
from langchain_core.documents import Document
import re


def scraper_repo() -> list[Path]:
    py_files = []
    # path = "data/raw/vllm-0.10.1/"    # Padrão
    path = Path("data_test")            # Para testar a função
    for file in path.rglob("*.py"):
        py_files.append(file)

    md_files = []
    for file in path.rglob("*.md"):
        md_files.append(file)
    return md_files


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
    """Splits a Markdown file by sections, protecting code blocks,
    and indexes each chunk with its correct start_index in the original text."""
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
            splitter_start_index_in_section = doc.metadata.get("start_index", 0)

            # Posição absoluta do início do chunk no texto processado (com placeholders)
            absolute_splitter_index = section_start + splitter_start_index_in_section

            # 4. Calcular o delta acumulado considerando os blocos anteriores a esta posição absoluta
            delta_acumulado = 0
            for block in code_blocks:
                if block["start"] < (absolute_splitter_index + delta_acumulado):
                    delta_acumulado += block["len_diff"]

            # O start_index real no texto original completo
            real_start_index = absolute_splitter_index + delta_acumulado

            # 5. Restaurar os placeholders de volta para os blocos de código originais
            for block in code_blocks:
                if block["placeholder"] in content:
                    content = content.replace(block["placeholder"], block["original"])

            new_metadata = dict(doc.metadata)
            new_metadata["start_index"] = real_start_index

            final_documents.append(Document(page_content=content, metadata=new_metadata))

    return final_documents


if __name__ == "__main__":
    md_file = scraper_repo()
    with open(md_file[0]) as archive:
        content = archive.read()
    chunk = chunk_text(content)
    print(chunk)
