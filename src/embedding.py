from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    本地中文Embedding模型
    模型: BAAI/bge-small-zh-v1.5（512维）
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings
