from langchain_openai import ChatOpenAI
from config.settings import settings


def get_llm(temperature: float = 0.7):
    """
    DeepSeek LLM（兼容OpenAI格式）
    """
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
    return llm
