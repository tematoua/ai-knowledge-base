"""
Hello LangChain - 测试 DeepSeek LLM 连接
"""
import os
from langchain_deepseek import ChatDeepSeek

api_key = os.environ.get("DEEPSEEK_API_KEY") or input("请输入 DEEPSEEK_API_KEY: ")

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
)

response = llm.invoke("你好，请用一句话介绍自己")
print(response.content)
