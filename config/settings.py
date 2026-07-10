"""
配置管理模块
集中管理所有配置项，从环境变量读取
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    """应用配置"""
    
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    # 模型配置
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # 文本切分配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # 检索配置
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
    
    # Rerank配置（可选，使用LLM-based重排序）
    USE_RERANK = os.getenv("USE_RERANK", "false").lower() == "true"
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
    
    # 数据目录
    PAPERS_DIR = BASE_DIR / "data" / "papers"

# 全局配置实例
settings = Settings()
