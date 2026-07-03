import os
import bs4
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

# ========== 1. 加载网页文档 ==========
print("加载文档...")

def load_web_page(url: str) -> list[Document]:
    headers = {"User-Agent": "Mozilla/5.0"}  # 避免被拒
    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"无法访问 {url}，错误: {e}")
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    # 只取正文文本，忽略脚本和样式
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
docs = load_web_page(URL)
print(f"文档总字符数：{len(docs[0].page_content)}")

# ========== 2. 分割文档 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
all_splits = text_splitter.split_documents(docs)
print(f"切分成 {len(all_splits)} 个子文档")

# ========== 3. 加载 Embedding 模型（修正拼写） ==========
print("\n初始化 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # 修正：=v2 → -v2
    model_kwargs={"device": "cpu"},       # 如需 GPU 改为 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding 模型加载完成")

# ========== 4. 创建并持久化向量数据库 ==========
print("\n初始化向量数据库...")
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
# 显式持久化（部分版本需手动调用）
if hasattr(vector_store, "persist"):
    vector_store.persist()
print("向量数据库创建完成并已持久化")

# ========== 5. RAG 检索 + 生成 ==========
print("\n测试 RAG 问答...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
question = "What are the main components of an LLM agent?"

relevant_docs = retriever.invoke(question)
print(f"\n检索到 {len(relevant_docs)} 条相关文档：")
for i, doc in enumerate(relevant_docs, 1):
    print(f"  [{i}] {doc.page_content[:100]}...")

# ========== 6. 初始化 LLM（安全读取 API Key） ==========
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,   # 从环境变量读取，不再硬编码
    temperature=0.0,
)

# 构建提示（使用 HumanMessage 包装）
context = "\n\n".join([doc.page_content for doc in relevant_docs])
prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""

# 正确调用方式：传入消息列表
response = llm.invoke([HumanMessage(content=prompt)])
print(f"\nAI 回答:\n{response.content}")

print("\nRAG 流程完成！")