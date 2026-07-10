"""
RAG链模块 - Day 4 优化版
支持：引用来源、相似度过滤、Rerank、边界处理
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.vector_store import get_vector_store
from src.llm import get_llm
from config.settings import settings


def format_docs_with_source(docs: List[Document]) -> str:
    """
    格式化文档，包含来源信息
    
    格式:
        [相关段落 1] 来源: xxx.pdf 第3页
        内容...
    """
    if not docs:
        return "（未检索到相关内容）"
    
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        page_info = f" 第{page+1}页" if page != "" else ""
        
        formatted.append(
            f"[相关段落 {i+1}] 来源: {source}{page_info}\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted)


def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """
    简单的重排序：基于查询词在文档中的出现频率
    
    Day 4 简化版，后续可接入CrossEncoder模型
    """
    if not docs or not settings.USE_RERANK:
        return docs
    
    query_terms = set(query.lower().split())
    
    def score(doc):
        content = doc.page_content.lower()
        # 计算查询词命中次数
        hits = sum(1 for term in query_terms if term in content)
        # 结合原始相似度（如果有）
        base_score = doc.metadata.get("score", 0.5)
        return base_score + hits * 0.1
    
    # 排序并取Top-K
    ranked = sorted(docs, key=score, reverse=True)
    return ranked[:settings.RERANK_TOP_K]


# 优化后的Prompt模板
RAG_PROMPT_TEMPLATE = """你是一个专业的AI研究助手，擅长解释论文中的技术概念。

## 任务
基于以下检索到的论文内容，回答用户的问题。

## 检索到的相关内容
{context}

## 用户问题
{question}

## 回答规则
1. **只基于提供的上下文回答**，不要编造信息，不要引用外部知识
2. **如果上下文信息不足**，明确说明："根据提供的论文材料，我无法完整回答这个问题"
3. **回答要简洁清晰**，突出重点，避免冗长
4. **使用中文回答**，技术术语保留英文原文（如 Attention, Transformer）
5. **适当引用来源**：在关键观点后标注 [来自: xxx.pdf]
6. **如果涉及多个论文**，对比说明它们的不同观点

## 回答格式
直接给出回答，不需要重复问题。如果信息来自多个段落，整合后给出连贯回答。

回答："""


# 边界情况处理的系统提示
SYSTEM_PROMPT = """你是一个严谨的学术助手。在回答问题时：
- 明确区分事实和推测
- 如果信息来自论文，标注来源
- 如果信息不确定，明确说明"不确定"
- 拒绝回答与提供材料无关的问题"""


def create_rag_chain(temperature: float = 0.7, top_k: int = None):
    """
    创建优化的RAG链
    
    Args:
        temperature: LLM温度参数
        top_k: 检索数量，默认从配置读取
    
    Returns:
        可执行的RAG链
    """
    # 1. 准备组件
    vector_store = get_vector_store()
    top_k = top_k or settings.RETRIEVAL_TOP_K
    
    # 2. 配置Retriever（支持相似度分数）
    retriever = vector_store.db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD
        }
    )
    
    llm = get_llm(temperature=temperature)
    
    # 3. 构建Prompt
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # 4. 检索+重排序（可选）
    def retrieve_and_rerank(query: str):
        docs = retriever.invoke(query)
        return rerank_documents(query, docs)
    
    # 5. 组装RAG链
    rag_chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank) | format_docs_with_source,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    return rag_chain


def ask(question: str, temperature: float = 0.7, top_k: int = None) -> str:
    """
    快捷方式：直接提问
    
    Args:
        question: 用户问题
        temperature: LLM温度
        top_k: 自定义检索数量
    
    Returns:
        LLM生成的回答
    """
    chain = create_rag_chain(temperature=temperature, top_k=top_k)
    response = chain.invoke(question)
    return response.content


def ask_with_sources(question: str, temperature: float = 0.7, top_k: int = None) -> dict:
    """
    提问并返回结构化结果（包含引用来源）
    
    Returns:
        {
            "answer": "回答内容",
            "sources": ["xxx.pdf 第3页", "yyy.pdf 第5页"]
        }
    """
    chain = create_rag_chain(temperature=temperature, top_k=top_k)
    response = chain.invoke(question)
    
    # 提取来源（简单解析）
    answer = response.content
    sources = []
    # 从回答中提取 [来自: xxx.pdf] 格式的引用
    import re
    source_matches = re.findall(r'\[来自:\s*([^\]]+)\]', answer)
    sources = list(set(source_matches))
    
    return {
        "answer": answer,
        "sources": sources
    }


# 测试代码
if __name__ == "__main__":
    question = "Transformer中的注意力机制是怎么计算的？"
    print(f"问题: {question}")
    
    # 测试普通问答
    answer = ask(question)
    print(f"\n回答:\n{answer}")
    
    # 测试带来源的问答
    print("\n--- 带来源的回答 ---")
    result = ask_with_sources(question)
    print(f"回答: {result['answer']}")
    print(f"来源: {result['sources']}")
