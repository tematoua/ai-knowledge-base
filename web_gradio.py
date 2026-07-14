"""
Web界面模块
使用Gradio构建AI论文问答Web界面

Gradio 消息格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

启动方式:
    python web_gradio.py

访问地址:
    http://localhost:7860
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
from pathlib import Path

# 确保src在Python路径中
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from typing import List, Dict

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import get_vector_store
from src.rag_chain import ask_with_sources
from src.builtin_kb import init_builtin_kb, get_builtin_kb_status, rebuild_builtin_kb
from config.settings import settings


# 自定义CSS
CUSTOM_CSS = """
/* 全局 */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif !important;
    background: #f5f5f7 !important;
}

/* 所有卡片统一样式 */
.gradio-group, .panel {
    background: #fff !important;
    border: 1px solid #eaecef !important;
    border-radius: 16px !important;
}

/* 标题 */
.header-container { text-align: center; padding: 24px 0 12px 0; }
.header-title {
    font-size: 1.7em !important; font-weight: 700 !important;
    color: #1a1a2e; margin-bottom: 2px !important; letter-spacing: -0.5px;
}
.header-subtitle { font-size: 0.9em; color: #9ca3af; }
.header-desc {
    font-size: 0.82em; color: #9ca3af; background: #fff;
    padding: 5px 16px; border-radius: 20px;
    display: inline-block; margin-top: 6px;
    border: 1px solid #eaecef;
}

/* 折叠面板 */
.accordion {
    background: #fff !important;
    border: 1px solid #eaecef !important;
    border-radius: 16px !important;
}
.accordion > .label-wrap {
    color: #1a1a2e !important; font-weight: 600 !important;
    font-size: 0.92em !important; padding: 8px 0 !important;
}

/* 状态框 & 索引结果 */
.status-box textarea, .index-result textarea {
    border-radius: 12px !important;
    border: 1px solid #eaecef !important;
    background: #fafbfc !important;
    padding: 12px !important;
}

/* 按钮 */
button, .gr-button {
    border-radius: 12px !important;
    transition: all 0.15s !important;
}
button.primary {
    background: #1a1a2e !important; border: none !important;
    font-weight: 600 !important; min-height: 44px !important;
}
button.primary:hover { background: #2d2d4a !important; }
button.secondary {
    background: #f5f5f7 !important; color: #555 !important;
    border: 1px solid #eaecef !important; font-weight: 500 !important;
}
button.secondary:hover { background: #eeeef2 !important; }

/* 输入行对齐修复 */
/* 让 Row 内的子元素垂直居中 */
.form > .gr-row, .gr-box > .gr-row {
    align-items: center !important;
}
/* 发送按钮 & 输入框对齐 */
.input-box {
    display: flex !important; align-items: center !important;
}
.input-box textarea, .input-box input {
    border-radius: 12px !important;
    border: 1.5px solid #e0e3e8 !important;
    background: #fafbfc !important;
    padding: 11px 14px !important;
    font-size: 13.5px !important;
    min-height: 44px !important;
    transition: border-color 0.2s !important;
    line-height: 22px !important;
}
.input-box textarea:focus, .input-box input:focus {
    border-color: #1a1a2e !important;
    box-shadow: 0 0 0 3px rgba(26,26,46,0.06) !important;
}

/* 聊天区 */
.chatbot {
    border: 1px solid #eaecef !important;
    border-radius: 16px !important;
    background: #fff !important;
}
.chatbot .message-wrap { border-radius: 14px !important; }

/* 文件上传区 */
.file-preview { border-radius: 12px !important; }
input[type="file"] { border-radius: 12px !important; }

/* 示例按钮 */
.examples-row .example-btn {
    background: #f8f9fb !important;
    border: 1px solid #eaecef !important;
    border-radius: 18px !important;
    padding: 5px 14px !important;
    font-size: 0.82em !important; color: #555 !important;
}
.examples-row .example-btn:hover { background: #f0f1f5 !important; }

/* 底部提示 */
.footer-hint {
    text-align: center; color: #b0b0c0; font-size: 0.76em;
    margin-top: 10px; padding: 10px;
    background: #fff; border-radius: 16px;
    border: 1px solid #f0f1f5;
}

/* 分割线 */
hr.divider { border: none; height: 1px; background: #f0f1f5; margin: 14px 0; }
"""


# 全局状态
vector_store = get_vector_store()


# 状态管理
class LoadingState:
    """简单的加载状态管理"""
    def __init__(self):
        self.is_loading = False
        self.message = ""
    
    def start(self, message="处理中..."):
        self.is_loading = True
        self.message = message
    
    def stop(self):
        self.is_loading = False
        self.message = ""

loading_state = LoadingState()


def check_status():
    """检查系统状态（内置知识库 + 用户上传）"""
    kb_status = get_builtin_kb_status()

    try:
        vector_store.load()
    except Exception:
        # 向量数据库未初始化
        if kb_status["paper_names"]:
            return f" **内置知识库**  待初始化\n\n 发现 {len(kb_status['paper_names'])} 篇论文\n 启动时将自动索引，无需手动操作"
        return " **未索引论文**\n\n请先上传论文文件进行索引！"

    # 构建状态信息
    status_text = ""

    # 内置知识库
    if kb_status["initialized"]:
        status_text += " **内置知识库**  已就绪\n"
        status_text += f" 论文: {kb_status['papers_count']} 篇 | 块: {kb_status['chunks_count']}\n"
        if kb_status.get("pending_papers"):
            status_text += f" 新增待索引: {len(kb_status['pending_papers'])} 篇 (点击「重建知识库」)\n"
    elif kb_status["paper_names"]:
        status_text += f" **内置知识库** 待初始化 ({len(kb_status['paper_names'])} 篇)\n"
    else:
        status_text += " **内置知识库** 暂无内置论文\n"

    status_text += "\n"

    # 用户上传论文
    try:
        all_docs = vector_store.db._collection.get()
        metadatas = all_docs.get("metadatas", [])

        builtin_sources = {}
        user_sources = {}
        for meta in metadatas:
            if meta and "source" in meta:
                src = meta["source"]
                if meta.get("source_type") == "builtin":
                    builtin_sources[src] = builtin_sources.get(src, 0) + 1
                else:
                    user_sources[src] = user_sources.get(src, 0) + 1

        status_text += f" **用户上传** ({len(user_sources)} 篇)\n"
        if user_sources:
            for src, count in sorted(user_sources.items()):
                status_text += f"   • {src} ({count} 块)\n"
        else:
            status_text += "   （暂无用户上传）\n"

    except Exception as e:
        status_text += f"\n 统计异常: {e}"

    status_text += f"\n chunk={settings.CHUNK_SIZE} | Top-K={settings.RETRIEVAL_TOP_K} | 阈值={settings.RETRIEVAL_SCORE_THRESHOLD}"
    if kb_status.get("last_init"):
        status_text += f"\n 初始化: {kb_status['last_init'][:19]}"

    return status_text


def index_file(file_obj):
    """
    索引上传的文件
    
    Args:
        file_obj: Gradio上传的文件对象
    
    Returns:
        状态消息
    """
    if file_obj is None:
        return "请先选择文件"
    
    # Gradio file_obj 可能是路径字符串或NamedTuple
    if isinstance(file_obj, str):
        file_path = Path(file_obj)
    else:
        # Gradio 4.0+ 返回的是文件路径
        file_path = Path(file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
    
    if not file_path.exists():
        return f"文件不存在: {file_path}"
    
    try:
        # 1. 加载文档
        documents = load_document(file_path)
        
        # 2. 切分文档
        chunks = split_documents(documents)
        
        # 添加来源信息
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata.get("page_number", 0)
        
        # 3. 存入向量数据库（增量或新建）
        if not vector_store.is_initialized():
            try:
                vector_store.load()
            except Exception:
                pass
        
        if vector_store.is_initialized():
            vector_store.add_documents(chunks)
            mode = "增量添加"
        else:
            vector_store.create_from_documents(chunks)
            mode = "新建索引"
        
        return (
            f"索引成功！\n\n"
            f"文件: {file_path.name}\n"
            f"页数: {len(documents)}\n"
            f"文本块: {len(chunks)}\n"
            f"模式: {mode}"
        )
        
    except Exception as e:
        return f"索引失败: {str(e)}"


def chat(message: str, history: List[Dict]) -> str:
    """
    聊天问答
    
    Args:
        message: 用户问题
        history: 聊天历史 (Gradio 6.x格式: [{"role": "user"/"assistant", "content": "..."}])
    
    Returns:
        回答文本
    """
    # 检查是否有索引，无索引时尝试自动初始化
    if not vector_store.is_initialized():
        try:
            vector_store.load()
        except Exception:
            # 尝试自动初始化内置知识库
            kb_status = get_builtin_kb_status()
            if kb_status.get("paper_names"):
                init_result = init_builtin_kb()
                if init_result["status"] == "success":
                    try:
                        vector_store.load()
                    except Exception:
                        pass
                else:
                    return (
                        "知识库未初始化\n\n"
                        f"{init_result['message']}\n\n"
                        "请在左侧上传论文或手动运行: python cli.py init"
                    )
            else:
                return (
                    "请先索引论文！\n\n"
                    "请在左侧上传PDF或TXT文件，或添加论文到 data/papers/ 目录。"
                )
    
    try:
        # 调用RAG链（带引用来源）
        result = ask_with_sources(message, temperature=0.3, top_k=5)
        
        answer = result["answer"]
        sources = result["sources"]
        
        # 格式化回答（添加来源）
        if sources:
            answer += "\n\n---\n引用来源：\n"
            for i, source in enumerate(sources, 1):
                answer += f"{i}. {source}\n"
        
        return answer
        
    except Exception as e:
        return f"回答失败: {str(e)}"


def auto_init_kb():
    """启动时自动初始化内置知识库"""
    if not settings.BUILTIN_KB_ENABLED:
        print("内置知识库已禁用 (BUILTIN_KB_ENABLED=false)")
        return {"status": "disabled"}

    kb_status = get_builtin_kb_status()
    if kb_status["initialized"]:
        print(f"内置知识库已就绪: {kb_status['papers_count']} 篇论文, {kb_status['chunks_count']} 块")
        return kb_status
    if not kb_status["paper_names"]:
        print("内置知识库目录为空，跳过初始化")
        return kb_status

    if not settings.BUILTIN_KB_AUTO_INIT:
        print(f"⏸自动初始化已禁用 (BUILTIN_KB_AUTO_INIT=false), 发现 {len(kb_status['paper_names'])} 篇待索引论文")
        print(f" 手动初始化: python cli.py init")
        return kb_status

    force = settings.BUILTIN_KB_FORCE_INIT
    print(f"正在自动初始化内置知识库 ({len(kb_status['paper_names'])} 篇)..." + (" [强制模式]" if force else ""))
    result = init_builtin_kb(force=force)
    if result["status"] == "success":
        print(f"{result['message']}")
    else:
        print(f"{result['message']}")
    return result


def respond(message: str, history: List[Dict]) -> tuple:
    """
    处理用户消息并更新聊天历史
    
    Gradio格式: history = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        ("", updated_history)  # 清空输入框，更新聊天记录
    """
    if not message.strip():
        return "", history
    
    # 确保 history 是 list
    if history is None:
        history = []
    
    # 获取AI回答
    bot_message = chat(message, history)
    
    # Gradio: 追加消息到历史
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_message})
    
    return "", history


# Gradio 界面

def create_ui():
    """创建Gradio界面"""
    
    # 自定义主题
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="slate",
    )
    
    with gr.Blocks(
        title="AI论文智能问答助手",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:
        
        # 标题区域
        gr.HTML("""
        <div class="header-container">
            <div class="header-title">🤖 AI论文智能问答助手</div>
            <div class="header-subtitle">基于RAG技术，让论文阅读更高效</div>
            <div class="header-desc">📤 支持上传论文 → 🔍 自动索引 → 💬 直接提问</div>
        </div>
        """)
        
        with gr.Row(equal_height=False):
            # 左侧：上传和状态
            with gr.Column(scale=1, min_width=320):
                
                # 论文管理卡片
                gr.HTML('<div class="side-card-title">📚 论文管理</div>')
                
                with gr.Group(elem_classes=["side-card"]):
                    file_input = gr.File(
                        label="📎 上传论文文件",
                        file_types=[".pdf", ".txt", ".md"],
                        type="filepath"
                    )
                    
                    index_btn = gr.Button(
                        "🚀 开始索引",
                        variant="primary",
                        size="lg"
                    )
                    
                    index_result = gr.Textbox(
                        label="索引结果",
                        lines=5,
                        interactive=False,
                        elem_classes=["index-result"]
                    )
                
                gr.HTML('<hr class="divider">')
                
                # 系统状态卡片（可折叠，默认收起）
                with gr.Accordion("📊 系统状态", open=False, elem_classes=["side-card"]):
                    status_box = gr.Textbox(
                        label="",
                        lines=12,
                        interactive=False,
                        value=check_status(),
                        elem_classes=["status-box"]
                    )
                    
                    with gr.Row():
                        refresh_btn = gr.Button(
                            "🔄 刷新状态",
                            variant="secondary",
                            size="sm"
                        )
                        rebuild_kb_btn = gr.Button(
                            "🔁 重建知识库",
                            variant="secondary",
                            size="sm"
                        )
                
                        # 底部提示
                gr.HTML("""
                <div class="footer-hint">
                    📚 内置知识库: 将论文放入 data/papers/ 即可自动索引<br>
                    📤 用户上传: 通过上方上传按钮增量添加
                </div>
                """)
            
            # 右侧：聊天界面
            with gr.Column(scale=2):
                gr.HTML('<div class="side-card-title">💬 论文问答</div>')
                
                # 聊天机器人
                chatbot = gr.Chatbot(
                    height=520,
                    elem_classes=["chat-container"],
                    avatar_images=(
                        None,  # 用户头像使用默认
                        "https://api.dicebear.com/7.x/bottts/svg?seed=ai-assistant"  # AI头像
                    )
                )
                
                # 输入区域
                with gr.Row(equal_height=True):
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="输入你的问题，例如：Transformer中的注意力机制是怎么计算的？",
                        lines=1,
                        scale=8,
                        elem_classes=["input-box"],
                        show_label=False
                    )
                    send_btn = gr.Button(
                        "发送 ➤",
                        variant="primary",
                        scale=1,
                        min_width=80
                    )
                
                # 按钮行
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")
                    gr.Button("📋 复制最后回答", variant="secondary", size="sm", interactive=False)
                
                # 示例问题
                gr.HTML('<div style="margin-top:12px; font-size:0.9em; color:#666; font-weight:500;">📝 试试这些问题：</div>')
                
                examples = gr.Examples(
                    examples=[
                        "这篇论文的主要贡献是什么？",
                        "解释Transformer中的注意力机制",
                        "Transformer相比RNN有什么优势？",
                        "实验结果和结论是什么？",
                        "论文中的方法有哪些局限性？",
                    ],
                    inputs=msg_input,
                    examples_per_page=5,
                )
        
        # 事件绑定
        
        # 索引按钮
        index_btn.click(
            fn=index_file,
            inputs=[file_input],
            outputs=[index_result]
        ).then(
            fn=check_status,
            outputs=[status_box]
        )
        
        # 刷新状态
        refresh_btn.click(
            fn=check_status,
            outputs=[status_box]
        )

        # 重建知识库
        rebuild_kb_btn.click(
            fn=lambda: (rebuild_builtin_kb(), check_status()),
            outputs=[index_result, status_box]
        )
        
        # 发送按钮
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        # 回车发送
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        # 清空按钮
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input]
        )
    
    return demo


# 启动入口

# 首次启动: 自动初始化内置知识库
print("\n" + "=" * 56)
print("  🤖 AI论文智能问答助手 — 启动中")
print("=" * 56)
auto_init_kb()
print("=" * 56 + "\n")

if __name__ == "__main__":
    demo = create_ui()
    
    # 启动参数
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        quiet=False,
    )
