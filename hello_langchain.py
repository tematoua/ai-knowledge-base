from langchain_deepseek import ChatDeepSeek

# 初始化模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8858b069139141b99055010fc5f9467c"
)

# 第一个对话
response = llm.invoke("你好，请用一句话介绍自己")
print(response.content)
