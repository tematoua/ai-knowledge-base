from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
