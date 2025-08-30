from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from . import config


def get_chat(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure environment or .env file.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=config.OPENAI_API_KEY)


def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure environment or .env file.")
    return OpenAIEmbeddings(model=model, api_key=config.OPENAI_API_KEY)

