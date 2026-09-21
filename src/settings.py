"""Общие настройки DeepSeek для обоих шагов урока."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"


def create_deepseek_model() -> ChatOpenAI:
    """Создать LangChain-модель для OpenAI-совместимого DeepSeek API."""
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if not api_key or api_key == "put_your_key_here":
        raise RuntimeError(
            "Добавьте выданный преподавателем DEEPSEEK_API_KEY в файл .env."
        )

    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        use_responses_api=False,
        temperature=0,
        timeout=30.0,
        max_retries=0,
    )
