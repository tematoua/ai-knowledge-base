#!/usr/bin/env python
"""
支持：批量索引、增量添加、索引状态查看、检索测试
"""
import sys
from pathlib import Path

# 确保src在Python路径中
sys.path.insert(0, str(Path(__file__).parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import get_vector_store
from src.rag_chain import ask, ask_with_sources
from src.builtin_kb import init_builtin_kb, get_builtin_kb_status, rebuild_builtin_kb
from config.settings import settings

console = Console()


def print_banner():
    """打印欢迎横幅"""
    banner = Text()
    banner.append(" ", style="bold cyan")
    banner.append("AI论文智能问答助手\n", style="bold white")
    banner.append("基于RAG技术，让论文阅读更高效", style="dim")
    console.print(Panel(banner, border_style="cyan"))


@click.group()
def cli():
    """AI论文智能问答助手 - 命令行工具"""
    pass


@cli.command()
@click.option('--file', '-f', help='要索引的论文文件路径（PDF/TXT/Markdown）')
@click.option('--dir', '-d', help='要索引的目录（索引该目录下所有支持的文件）')
@click.option('--reset', is_flag=True, help='清空已有索引，重新创建')
@click.option('--show-chunks', is_flag=True, help='显示切分后的文本块示例')
def index(file: str, dir: str, reset: bool, show_chunks: bool):
    """
    索引论文文件到向量数据库
    
    示例:
        python cli.py index -f data/papers/attention.pdf
        python cli.py index -d data/papers/           # 索引整个目录
        python cli.py index -f data/papers/new.pdf    # 增量添加
        python cli.py index -d data/papers/ --reset   # 重建索引
    """
    print_banner()
    
    # 收集要处理的文件
    files_to_process = []
    
    if file:
        files_to_process.append(Path(file))
    elif dir:
        dir_path = Path(dir)
        if not dir_path.exists():
            console.print(f"[red] 目录不存在: {dir_path}[/red]")
            return
        # 收集所有支持的文件
        for ext in ['.pdf', '.txt', '.md', '.markdown']:
            files_to_process.extend(dir_path.glob(f'*{ext}'))
    else:
        console.print("[yellow] 请指定 --file 或 --dir[/yellow]")
        return
    
    if not files_to_process:
        console.print("[yellow] 没有找到可索引的文件[/yellow]")
        return
    
    console.print(f"\n[blue] 找到 {len(files_to_process)} 个文件待索引[/blue]")
    for f in files_to_process:
        console.print(f"  • {f.name}")
    
    # 加载向量数据库
    vector_store = get_vector_store()
    
    if reset:
        console.print("\n[yellow] 清空已有索引...[/yellow]")
        # 删除旧数据库
        import shutil
        db_path = Path(settings.CHROMA_PERSIST_DIRECTORY)
        if db_path.exists():
            shutil.rmtree(db_path)
        console.print("[green] 旧索引已清除[/green]")
    
    # 尝试加载已有数据库（非重置模式）
    if not reset:
        try:
            vector_store.load()
            console.print("[green]✓ 已加载现有索引[/green]")
        except Exception:
            console.print("[dim] 创建新索引...[/dim]")
    
    # 处理每个文件
    total_chunks = 0
    all_chunks = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for file_path in files_to_process:
            task = progress.add_task(f" {file_path.name}...", total=None)
            
            try:
                # 1. 加载文档
                documents = load_document(file_path)
                progress.update(task, description=f" {file_path.name} - 已加载 {len(documents)} 页")
                
                # 2. 切分文档
                chunks = split_documents(documents)
                
                # 添加来源信息到metadata
                for chunk in chunks:
                    chunk.metadata["source"] = file_path.name
                    # 确保page信息存在
                    if "page" not in chunk.metadata:
                        chunk.metadata["page"] = chunk.metadata.get("page_number", 0)
                
                all_chunks.extend(chunks)
                total_chunks += len(chunks)
                
                progress.update(task, description=f" {file_path.name} -  {len(chunks)} 块")
                
                # 显示切分示例
                if show_chunks and chunks:
                    console.print(f"\n[dim]  切分示例（前200字符）:[/dim]")
                    console.print(f"  [dim]{chunks[0].page_content[:200]}...[/dim]")
                
            except Exception as e:
                console.print(f"\n[red]❌ 处理失败 {file_path.name}: {str(e)}[/red]")
                continue
    
    # 存入向量数据库
    if all_chunks:
        console.print(f"\n[blue] 正在索引 {len(all_chunks)} 个文本块...[/blue]")
        try:
            if vector_store.is_initialized():
                # 增量添加
                vector_store.add_documents(all_chunks)
                console.print("[green] 增量添加完成[/green]")
            else:
                # 新建索引
                vector_store.create_from_documents(all_chunks)
                console.print("[green] 新索引创建完成[/green]")
            
            console.print(f"\n[bold green] 索引完成！[/bold green]")
            console.print(f"  文件数: {len(files_to_process)}")
            console.print(f"  文本块: {total_chunks}")
            console.print(f"  数据库: {settings.CHROMA_PERSIST_DIRECTORY}")
            
        except Exception as e:
            console.print(f"\n[red] 索引失败: {str(e)}[/red]")
            raise


@cli.command()
@click.option('--force', '-f', is_flag=True, help='强制重建内置知识库')
def init(force: bool):
    """
    初始化内置知识库
    
    扫描 data/papers/ 目录，自动索引所有论文到向量数据库。
    无需手动指定文件，一键完成。
    
    示例:
        python cli.py init           # 初始化（已有则跳过）
        python cli.py init --force   # 强制重建
    """
    print_banner()

    # 先检查状态
    kb_status = get_builtin_kb_status()

    console.print(f"\n[bold]📚 内置知识库状态[/bold]")
    console.print(f"  论文目录: {settings.PAPERS_DIR}")
    console.print(f"  论文数: {kb_status['papers_count']}")

    if kb_status["initialized"] and not force:
        console.print(f"\n[green]✅ 内置知识库已就绪[/green]")
        if kb_status.get("pending_papers"):
            console.print(f"[yellow]🆕 检测到 {len(kb_status['pending_papers'])} 篇新论文未索引[/yellow]")
            console.print(f"[dim]   使用 --force 重建可包含新论文[/dim]")
        return

    if not kb_status["paper_names"]:
        console.print("\n[yellow]📭 data/papers/ 目录为空[/yellow]")
        console.print(f"[dim]   请将论文PDF放入: {settings.PAPERS_DIR}[/dim]")
        return

    # 开始初始化
    console.print(f"\n[blue]🔧 正在初始化内置知识库 ({len(kb_status['paper_names'])} 篇论文)...[/blue]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        def progress_cb(stage, info):
            if stage == "start":
                progress.add_task(f"准备索引 {info['total']} 篇论文...", total=None)
            elif stage == "clearing":
                progress.add_task("清空旧索引...", total=None)
            elif stage == "indexing":
                progress.add_task(
                    f"[{info['current']}/{info['total']}] {info['name']}...",
                    total=None
                )

        result = init_builtin_kb(force=force, progress_callback=progress_cb)

    console.print(f"\n[bold]{result['message']}[/bold]")

    if result.get("paper_names"):
        console.print("\n[bold]📄 已索引论文:[/bold]")
        for name in result["paper_names"]:
            console.print(f"  [green]• {name}[/green]")

    if result.get("errors"):
        console.print("\n[bold yellow]⚠️ 跳过的文件:[/bold yellow]")
        for e in result["errors"]:
            console.print(f"  [yellow]  • {e['file']}: {e['error']}[/yellow]")


@cli.command()
@click.argument('question')
@click.option('--temperature', '-t', default=0.7, help='LLM温度参数 (0.0-1.0)')
@click.option('--top-k', '-k', default=None, type=int, help='检索数量（覆盖默认配置）')
@click.option('--show-sources', '-s', is_flag=True, help='显示引用来源')
def ask_cmd(question: str, temperature: float, top_k: int, show_sources: bool):
    """
    向已索引的论文提问
    
    示例:
        python cli.py ask "这篇论文的主要贡献是什么？"
        python cli.py ask "解释注意力机制" -t 0.3 -k 5
        python cli.py ask "对比两篇论文的观点" -s
    """
    print_banner()
    
    vector_store = get_vector_store()
    
    # 检查是否有索引
    if not vector_store.is_initialized():
        try:
            vector_store.load()
        except Exception:
            console.print("[red] 没有找到索引数据！请先运行: python cli.py index -d data/papers/[/red]")
            return
    
    console.print(f"\n[blue] 问题: {question}[/blue]")
    console.print(f"[dim]参数: temperature={temperature}, top_k={top_k or settings.RETRIEVAL_TOP_K}[/dim]")
    console.print("[dim]正在检索并生成回答...[/dim]\n")
    
    try:
        if show_sources:
            # 使用带来源的回答
            result = ask_with_sources(question, temperature=temperature, top_k=top_k)
            
            console.print(Panel(
                result["answer"],
                title="[bold green]回答[/bold green]",
                border_style="green"
            ))
            
            if result["sources"]:
                console.print("\n[bold blue] 引用来源:[/bold blue]")
                for source in result["sources"]:
                    console.print(f"  • {source}")
        else:
            # 普通回答
            answer = ask(question, temperature=temperature, top_k=top_k)
            console.print(Panel(
                answer,
                title="[bold green]回答[/bold green]",
                border_style="green"
            ))
        
    except Exception as e:
        console.print(f"[red] 回答失败: {str(e)}[/red]")
        raise


@cli.command()
@click.argument('question')
@click.option('--k', default=5, help='检索数量')
def search(question: str, k: int):
    """
    测试检索效果（不调用LLM，直接看检索结果）
    
    示例:
        python cli.py search "注意力机制" -k 5
    """
    print_banner()
    
    vector_store = get_vector_store()
    
    if not vector_store.is_initialized():
        try:
            vector_store.load()
        except Exception:
            console.print("[red] 没有索引数据！[/red]")
            return
    
    console.print(f"\n[blue]🔍 检索: {question}[/blue]")
    console.print(f"[dim]Top-K: {k}[/dim]\n")
    
    try:
        results = vector_store.similarity_search(question, k=k)
        
        if not results:
            console.print("[yellow] 未找到相关内容[/yellow]")
            return
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            page_str = f" 第{page+1}页" if page != "" else ""
            
            console.print(Panel(
                f"[dim]{doc.page_content[:500]}...[/dim]",
                title=f"[bold cyan]结果 {i}[/bold cyan] [dim]{source}{page_str}[/dim]",
                border_style="cyan"
            ))
        
        console.print(f"\n[green] 找到 {len(results)} 个相关段落[/green]")
        
    except Exception as e:
        console.print(f"[red] 检索失败: {str(e)}[/red]")


@cli.command()
def status():
    """查看系统状态和索引统计"""
    print_banner()
    
    # 配置信息表格
    config_table = Table(title="系统配置", border_style="blue")
    config_table.add_column("配置项", style="cyan")
    config_table.add_column("值", style="green")
    
    config_table.add_row("向量数据库", str(settings.CHROMA_PERSIST_DIRECTORY))
    config_table.add_row("LLM模型", settings.LLM_MODEL)
    config_table.add_row("Embedding模型", settings.EMBEDDING_MODEL)
    config_table.add_row("文本块大小", str(settings.CHUNK_SIZE))
    config_table.add_row("块重叠", str(settings.CHUNK_OVERLAP))
    config_table.add_row("检索Top-K", str(settings.RETRIEVAL_TOP_K))
    config_table.add_row("相似度阈值", str(settings.RETRIEVAL_SCORE_THRESHOLD))
    config_table.add_row("启用Rerank", "✓" if settings.USE_RERANK else "✗")
    
    console.print(config_table)
    
    # === 内置知识库状态 ===
    kb_status = get_builtin_kb_status()
    
    console.print("\n[bold]📚 内置知识库[/bold]")
    console.print(f"  论文目录: {settings.PAPERS_DIR}")
    if kb_status["initialized"]:
        console.print(f"  [green]状态: ✅ 已就绪[/green]")
        console.print(f"  论文数: {kb_status['papers_count']}")
        console.print(f"  文本块: {kb_status['chunks_count']}")
        if kb_status.get("last_init"):
            console.print(f"  初始化时间: {kb_status['last_init'][:19]}")
    else:
        console.print(f"  [yellow]状态: 🟡 未初始化[/yellow]")
        console.print(f"  待索引论文: {len(kb_status['paper_names'])} 篇")
        console.print(f"  [dim]运行 'python cli.py init' 初始化[/dim]")
    
    if kb_status.get("paper_names"):
        console.print("\n  [bold]论文列表:[/bold]")
        for name in kb_status["paper_names"]:
            indexed = "✅" if name in kb_status.get("indexed_papers", []) else "🆕"
            console.print(f"    {indexed} {name}")
    
    if kb_status.get("pending_papers"):
        console.print(f"\n  [yellow]🆕 新增待索引: {len(kb_status['pending_papers'])} 篇[/yellow]")
    
    # === 用户上传状态 ===
    vector_store = get_vector_store()
    console.print("\n[bold]📁 用户上传论文[/bold]")
    try:
        vector_store.load()
        try:
            collection = vector_store.db._collection
            all_docs = collection.get()
            metadatas = all_docs.get("metadatas", [])
            
            user_sources = {}
            for meta in metadatas:
                if meta and "source" in meta and meta.get("source_type") != "builtin":
                    src = meta["source"]
                    user_sources[src] = user_sources.get(src, 0) + 1
            
            if user_sources:
                for src, count in sorted(user_sources.items()):
                    console.print(f"  [green]• {src} ({count} 块)[/green]")
            else:
                console.print("  [dim]（暂无用户上传）[/dim]")
                
        except Exception as e:
            console.print(f"[dim]  无法获取详细统计: {e}[/dim]")
            
    except Exception:
        console.print("  [yellow]向量数据库未初始化[/yellow]")


if __name__ == "__main__":
    cli()
