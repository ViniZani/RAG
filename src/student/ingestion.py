from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def scraper_repo() -> list[Path]:
    py_files = []
    # path = "data/raw/vllm-0.10.1/"
    path = Path("data_test")
    for file in path.rglob("*.py"):
        py_files.append(file)

    md_files = []
    for file in path.rglob("*.md"):
        md_files.append(file)
    return md_files


# def chunk_python_code(text: str, max_chunk_size: int) -> list[...]:
    # ast
    # pass


def chunk_text(text: str, max_chunk_size: int = 2000) -> list[Document]:
    """Splits a Markdown file by chunks
    and index each chunk by your own start_index"""""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    splitter = MarkdownTextSplitter(chunk_size=max_chunk_size,
                                    chunk_overlap=0,
                                    add_start_index=True,)
    return splitter.create_documents([text])


if __name__ == "__main__":
    md_file = scraper_repo()
    with open(md_file[0]) as archive:
        content = archive.read()
    chunk = chunk_text(content)
    print(chunk)
