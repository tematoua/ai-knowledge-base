from typing import List, Optional
from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from src.embedding import get_embedding_model
from config.settings import settings


class VectorStore:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.embedding_model = get_embedding_model()
        self.db: Optional[Chroma] = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        # Chroma >= 0.5 auto-persists; no need for manual persist()
        return self.db

    def load(self) -> Chroma:
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )
        return self.db

    def add_documents(self, documents: List[Document]):
        if self.db is None:
            self.create_from_documents(documents)
        else:
            self.db.add_documents(documents)
            # Chroma >= 0.5 auto-persists

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if self.db is None:
            raise ValueError("向量数据库未初始化")
        k = k or settings.RETRIEVAL_TOP_K
        return self.db.similarity_search(query, k=k)

    def is_initialized(self) -> bool:
        return self.db is not None


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
