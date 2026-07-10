from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_txt(file_path: Union[str, Path]) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(file_path), encoding='utf-8')
    return loader.load()


def load_markdown(file_path: Union[str, Path]) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    loader = UnstructuredMarkdownLoader(str(file_path))
    return loader.load()


def load_document(file_path: Union[str, Path]) -> List[Document]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return load_pdf(file_path)
    elif suffix == '.txt':
        return load_txt(file_path)
    elif suffix in ['.md', '.markdown']:
        return load_markdown(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")
