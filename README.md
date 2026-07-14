# 🤖 AI论文智能问答助手

基于RAG（检索增强生成）技术的AI论文问答系统。支持PDF论文索引、智能检索、引用来源标注。

## ✨ 功能特性

- 📄 **多格式支持**：PDF / TXT / Markdown
- 🔍 **智能检索**：相似度阈值过滤、Top-K可调
- 📚 **多论文联合索引**：支持同时索引多篇论文，跨论文问答
- 📝 **引用来源标注**：回答自动标注引用页码
- 🛡️ **边界处理**：信息不足时明确说明，拒绝编造
- 🖥️ **Web界面**：Gradio交互式界面，拖拽上传、聊天问答
- 🖥️ **命令行工具**：批量索引、增量添加、检索调试

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| RAG框架 | LangChain + LCEL |
| 向量数据库 | Chroma |
| Embedding | BAAI/bge-small-zh-v1.5（本地模型，免费） |
| LLM | DeepSeek API（deepseek-chat） |
| CLI | Click + Rich |
| Web界面 | Gradio |

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
├── web_gradio.py             # Web界面（Gradio）
├── config/settings.py        # 配置管理
├── data/papers/              # 论文存放目录
├── chroma_db/                # 向量数据库（自动创建）
├── requirements.txt
├── .env                      # 环境变量（不提交git）
└── .env.example              # 环境变量模板
```

## 🚀 快速开始

### 🌐 Web 界面（推荐！最简单）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 3. 启动 Web 服务
python web_gradio.py

# 4. 浏览器打开 http://localhost:7860
```

功能：
- 📚 **上传论文**：拖拽PDF/TXT文件，自动索引
- 💬 **聊天问答**：输入问题，获取带引用来源的回答
- 📊 **系统状态**：查看已索引论文和配置信息

### 💻 命令行工具（高级用户）

```bash
# 1. 安装依赖
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 3. 准备论文
mkdir -p data/papers
cp your_paper.pdf data/papers/

# 4. 索引论文
python cli.py index -d data/papers/

# 5. 提问
python cli.py ask "Transformer中的注意力机制是怎么计算的？" -s
```

## 🖥️ Web 界面使用指南

### 启动方式

```bash
# 本地启动（http://localhost:7860）
python web_gradio.py
```

### 使用步骤

1. **上传论文**：左侧拖拽或选择PDF/TXT文件，点击「索引文件」
2. **等待索引**：显示索引成功提示（约几秒到几分钟）
3. **开始问答**：右侧输入问题，按回车或点击发送
4. **查看回答**：AI回答带引用来源标注，如「[来自: xxx.pdf 第3页]」

### 界面布局

| 区域 | 功能 |
|------|------|
| 左侧上传区 | 文件上传、索引按钮、系统状态 |
| 右侧聊天区 | 消息输入、历史记录、示例问题 |

## 🖥️ CLI 命令详细说明

### 索引命令

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

## 📝 更新日志

### Day 5 (2026-07-10~07-12)
- ✅ Web界面V1.0：Gradio交互式界面，拖拽上传、聊天问答
- ✅ Web整合RAG后端：复用Day 4的rag_chain，引用来源标注
- ✅ 界面布局：左侧上传/状态，右侧聊天问答

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
