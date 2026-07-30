# flake8: noqa
# type: none
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, PythonCodeTextSplitter
from langchain_community.vectorstores import Chroma
from src.student.ingestion import scraper_repo, chunk_text, chunk_python_code
from src.student.index import save_docs_vectordb


if __name__ == "__main__":
    md_file = scraper_repo()
    with open(md_file[0]) as archive:
        content = archive.read()
    chunk = chunk_text(content)
    save_docs_vectordb(chunk)
    print(chunk)