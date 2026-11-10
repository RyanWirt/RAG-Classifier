from typing import Any

from app.config import Settings


def create_chat_model(settings: Settings) -> Any:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model, temperature=settings.temperature
        )
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.chat_model, temperature=settings.temperature
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
