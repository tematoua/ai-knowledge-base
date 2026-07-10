# 🤖 AI论文智能问答助手

基于RAG（检索增强生成）技术的AI论文问答系统。支持PDF论文索引、智能检索、引用来源标注。

## ✨ 功能特性

- 📄 **多格式支持**：PDF / TXT / Markdown
- 🔍 **智能检索**：相似度阈值过滤、Top-K可调
- 📚 **多论文联合索引**：支持同时索引多篇论文，跨论文问答
- 📝 **引用来源标注**：回答自动标注引用页码
- 🛡️ **边界处理**：信息不足时明确说明，拒绝编造
- 🖥️ **命令行工具**：批量索引、增量添加、检索调试

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| RAG框架 | LangChain + LCEL |
| 向量数据库 | Chroma |
| Embedding | BAAI/bge-small-zh-v1.5（本地模型，免费） |
| LLM | DeepSeek API（deepseek-chat） |
| CLI | Click + Rich |

## 📁 项目结构

```
ai-knowledge-base/
├── src/
│   ├── document_loader.py    # 文档加载（PDF/TXT/Markdown）
│   ├── text_splitter.py      # 文本切分（RecursiveCharacterTextSplitter）
│   ├── embedding.py          # Embedding模型（BAAI/bge-small-zh-v1.5）
│   ├── vector_store.py       # 向量存储（Chroma）
│   ├── llm.py                # LLM封装（DeepSeek）
│   └── rag_chain.py          # RAG核心链（检索+生成+来源标注）
├── cli.py                    # 命令行入口
├── config/settings.py        # 配置管理
├── data/papers/              # 论文存放目录
├── chroma_db/                # 向量数据库（自动创建）
├── requirements.txt
├── .env                      # 环境变量（不提交git）
└── .env.example              # 环境变量模板
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd ai-knowledge-base

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 3. 准备论文

把论文PDF放到 `data/papers/` 目录：
```bash
mkdir -p data/papers
cp your_paper.pdf data/papers/
```

### 4. 索引论文

```bash
# 索引单篇论文
python cli.py index -f data/papers/attention.pdf

# 批量索引整个目录（推荐）
python cli.py index -d data/papers/

# 清空重建索引
python cli.py index -d data/papers/ --reset
```

### 5. 提问

```bash
# 基础提问
python cli.py ask "Transformer中的注意力机制是怎么计算的？"

# 低温度（更确定性）
python cli.py ask "解释多头注意力" -t 0.3

# 显示引用来源
python cli.py ask "Transformer相比RNN的优势？" -s

# 调整检索数量
python cli.py ask "注意力机制的应用" -k 5
```

### 6. 其他命令

```bash
# 检索调试（不调用LLM，直接看检索结果）
python cli.py search "注意力机制的优势" --k 3

# 查看系统状态和索引统计
python cli.py status
```

## ⚙️ 配置说明

所有配置通过环境变量管理，编辑 `.env` 文件即可：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CHUNK_SIZE` | 1000 | 文本块大小，越大上下文越完整 |
| `CHUNK_OVERLAP` | 100 | 块间重叠，保证连贯性 |
| `RETRIEVAL_TOP_K` | 5 | 检索返回数量 |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.3 | 相似度阈值，过滤低质量匹配 |
| `USE_RERANK` | false | 是否启用重排序 |

## 📊 效果对比

| 指标 | Day 3 | Day 4 | 提升 |
|------|-------|-------|------|
| 文本块大小 | 500 | 1000 | 上下文更完整 |
| 检索Top-K | 3 | 5 | 召回率更高 |
| 引用来源 | ❌ 无 | ✅ 自动标注 | 可信度提升 |
| 结构化回答 | ❌ 段落式 | ✅ 分点式 | 可读性提升 |
| 边界处理 | ❌ 编造 | ✅ 明确说明 | 安全性提升 |

## 🗓️ 阶段规划

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| 阶段1 | 7.3-7.8 | 基础构建：命令行RAG | ✅ 已完成 |
| 阶段2 | 7.9-7.15 | 平台实战：Web版V1.0 | 🟡 进行中 |
| 阶段3 | 7.16-7.22 | 工程化：Docker+Milvus+云部署 | ⏳ 待开始 |
| 阶段4 | 7.23-7.29 | 高阶能力：GraphRAG V2.0 | ⏳ 待开始 |
| 阶段5 | 7.30-8.5 | 求职准备：简历+面试 | ⏳ 待开始 |
| 阶段6 | 8.6-8.12 | 投递实战：拿Offer | ⏳ 待开始 |

## 📝 更新日志

### Day 4 (2026-07-10)
- ✅ 检索优化：chunk_size=1000, Top-K=5, 相似度阈值=0.3
- ✅ Prompt工程：结构化回答、引用来源标注、边界处理
- ✅ CLI增强：批量索引(-d)、增量添加、检索调试(search)、状态查看(status)
- ✅ 多论文联合索引：2篇论文同时索引，跨论文问答

### Day 3 (2026-07-09)
- ✅ 接入真实Embedding：BAAI/bge-small-zh-v1.5
- ✅ 接入真实LLM：DeepSeek API
- ✅ 端到端RAG跑通：提问→检索→生成回答

### Day 1-2 (2026-07-03~07-05)
- ✅ 项目骨架搭建
- ✅ PDF加载、文本切分、向量索引基础功能

## 📄 License

MIT
