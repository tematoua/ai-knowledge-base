"""
内置知识库管理模块

功能:
  1. 启动时自动扫描 data/papers/ 目录, 若无索引则自动构建
  2. 标记内置知识库内容 (source_type="builtin")
  3. 初始化状态管理 (避免重复索引)
  4. 支持强制重建 (--force)

使用:
  from src.builtin_kb import init_builtin_kb, get_builtin_kb_status
  result = init_builtin_kb()
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import get_vector_store
from config.settings import settings


# 内置知识库初始化标记文件
_INIT_FLAG_FILE = Path(settings.CHROMA_PERSIST_DIRECTORY) / ".builtin_kb_init.json"


def init_builtin_kb(force: bool = False, progress_callback=None) -> Dict:
    """
    初始化内置知识库: 扫描 data/papers/ 目录, 自动索引所有论文

    Args:
        force: 强制重建 (即使已有索引)
        progress_callback: 进度回调 fn(stage, info)

    Returns:
        {
            "status": "success" | "already_initialized" | "empty" | "error",
            "message": "描述信息",
            "papers": 论文数量,
            "chunks": 文本块数量,
            "paper_names": ["...pdf", ...]
        }
    """
    papers_dir = settings.PAPERS_DIR

    # 确保目录存在
    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有论文文件
    papers = []
    for ext in ['.pdf', '.txt', '.md', '.markdown']:
        papers.extend(sorted(papers_dir.glob(f'*{ext}')))

    if not papers:
        return {
            "status": "empty",
            "message": "📭 内置知识库目录为空, 请将论文 PDF 放入 data/papers/ 目录",
            "papers": 0,
            "chunks": 0,
            "paper_names": []
        }

    # 检查是否已初始化过
    if not force:
        # 方式1: 检查初始化标记文件
        if _INIT_FLAG_FILE.exists():
            try:
                flag_data = json.loads(_INIT_FLAG_FILE.read_text(encoding="utf-8"))
                # 检查论文是否有变化
                current_files = sorted([p.name for p in papers])
                if flag_data.get("papers") == current_files:
                    return {
                        "status": "already_initialized",
                        "message": f"✅ 内置知识库已就绪 ({len(papers)} 篇论文)",
                        "papers": len(papers),
                        "chunks": flag_data.get("chunks", 0),
                        "paper_names": current_files
                    }
            except Exception:
                pass

        # 方式2: 检查 Chroma DB 是否已有数据
        if _vector_store_has_data():
            return {
                "status": "already_initialized",
                "message": f"✅ 知识库索引已存在 ({len(papers)} 篇论文)",
                "papers": len(papers),
                "chunks": 0,  # 无法精确获取
                "paper_names": [p.name for p in papers]
            }

    # === 开始索引 ===
    if progress_callback:
        progress_callback("start", {"total": len(papers)})

    vector_store = get_vector_store()

    # 如果强制重建, 先清空
    if force and _vector_store_has_data():
        if progress_callback:
            progress_callback("clearing", {})
        import shutil
        db_path = Path(settings.CHROMA_PERSIST_DIRECTORY)
        if db_path.exists():
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    paper_names = []
    errors = []

    for i, paper_path in enumerate(papers):
        if progress_callback:
            progress_callback("indexing", {
                "current": i + 1,
                "total": len(papers),
                "name": paper_path.name
            })

        try:
            # 1. 加载
            documents = load_document(paper_path)

            # 2. 切分
            chunks = split_documents(documents)

            # 3. 标记元数据
            for chunk in chunks:
                chunk.metadata["source"] = paper_path.name
                chunk.metadata["source_type"] = "builtin"  # 标记为内置知识库
                if "page" not in chunk.metadata:
                    chunk.metadata["page"] = chunk.metadata.get("page_number", 0)

            all_chunks.extend(chunks)
            paper_names.append(paper_path.name)

        except Exception as e:
            errors.append({"file": paper_path.name, "error": str(e)})
            continue

    if not all_chunks:
        return {
            "status": "error",
            "message": f"❌ 没有成功索引任何论文\n错误: {errors}",
            "papers": 0,
            "chunks": 0,
            "paper_names": [],
            "errors": errors
        }

    # 4. 存入向量数据库
    try:
        vector_store.create_from_documents(all_chunks)

        # 5. 写入初始化标记
        _write_init_flag(paper_names, len(all_chunks))

        msg_parts = [f"✅ 内置知识库初始化完成"]
        msg_parts.append(f"📄 论文: {len(papers)} 篇")
        msg_parts.append(f"🧩 文本块: {len(all_chunks)} 个")
        if errors:
            msg_parts.append(f"⚠️ 跳过的文件: {len(errors)} 个")

        return {
            "status": "success",
            "message": "\n".join(msg_parts),
            "papers": len(papers),
            "chunks": len(all_chunks),
            "paper_names": paper_names,
            "errors": errors if errors else None
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ 向量数据库写入失败: {str(e)}",
            "papers": 0,
            "chunks": 0,
            "paper_names": paper_names,
            "errors": [{"error": str(e)}]
        }


def get_builtin_kb_status() -> Dict:
    """
    获取内置知识库状态 (不触发索引)

    Returns:
        {
            "initialized": True/False,
            "papers_count": 论文数,
            "chunks_count": 文本块数,
            "paper_names": ["...pdf", ...],
            "pending_papers": 未索引的论文列表
        }
    """
    papers_dir = settings.PAPERS_DIR

    # 目录中的论文
    current_papers = []
    for ext in ['.pdf', '.txt', '.md', '.markdown']:
        current_papers.extend(sorted(papers_dir.glob(f'*{ext}')))

    current_names = sorted([p.name for p in current_papers])

    # 检查初始化状态 (三层判断)
    is_initialized = _INIT_FLAG_FILE.exists()
    flag_data = {}
    if is_initialized:
        try:
            flag_data = json.loads(_INIT_FLAG_FILE.read_text(encoding="utf-8"))
            indexed_names = flag_data.get("papers", [])
            if sorted(indexed_names) != current_names:
                is_initialized = False  # 论文有变化
        except Exception:
            is_initialized = False

    # 兼容: 没有标记文件但 chroma_db 已有数据 (旧版索引)
    if not is_initialized:
        try:
            vs = get_vector_store()
            vs.load()
            count = vs.db._collection.count()
            if count > 0:
                is_initialized = True
                # 构造虚拟标记数据
                all_docs = vs.db._collection.get()
                metadatas = all_docs.get("metadatas", [])
                indexed = list(set(m.get("source", "") for m in metadatas if m))
                flag_data = {
                    "version": 0,
                    "init_time": "(旧版索引)",
                    "count": len(indexed),
                    "chunks": count,
                    "papers": sorted(indexed),
                    "config": {"note": "从旧版 chroma_db 恢复"}
                }
        except Exception:
            pass

    # 检查是否有新增论文
    pending_papers = []
    if is_initialized and flag_data:
        indexed_set = set(flag_data.get("papers", []))
        pending_papers = [n for n in current_names if n not in indexed_set]
    elif not is_initialized and current_names:
        pending_papers = current_names

    return {
        "initialized": is_initialized,
        "papers_count": flag_data.get("count", len(current_names)) if is_initialized else len(current_names),
        "chunks_count": flag_data.get("chunks", 0) if is_initialized else 0,
        "paper_names": current_names,
        "indexed_papers": flag_data.get("papers", []) if is_initialized else [],
        "pending_papers": pending_papers,
        "last_init": flag_data.get("init_time", ""),
        "init_version": flag_data.get("version", 0)
    }


def rebuild_builtin_kb(progress_callback=None) -> Dict:
    """强制重建内置知识库"""
    return init_builtin_kb(force=True, progress_callback=progress_callback)


def _vector_store_has_data() -> bool:
    """检查向量数据库是否已有数据"""
    try:
        vector_store = get_vector_store()
        vector_store.load()
        collection = vector_store.db._collection
        return collection.count() > 0
    except Exception:
        return False


def _write_init_flag(papers: List[str], chunks: int):
    """写入初始化标记文件"""
    flag_data = {
        "version": 1,
        "init_time": datetime.now().isoformat(),
        "count": len(papers),
        "chunks": chunks,
        "papers": sorted(papers),
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.EMBEDDING_MODEL
        }
    }
    _INIT_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INIT_FLAG_FILE.write_text(json.dumps(flag_data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # 快速测试
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=== 内置知识库状态 ===")
    status = get_builtin_kb_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n=== 初始化内置知识库 ===")
    result = init_builtin_kb()
    for k, v in result.items():
        print(f"  {k}: {v}")
