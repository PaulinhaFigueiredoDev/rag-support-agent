import os

from dotenv import load_dotenv

load_dotenv()


def get_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")


def get_rag_top_k() -> int:
    """Quantidade de trechos de política consultados por pergunta."""
    raw_value = os.getenv("RAG_TOP_K", "3").strip()
    try:
        top_k = int(raw_value)
    except ValueError:
        raise ValueError("RAG_TOP_K deve ser um inteiro entre 1 e 5") from None
    if not 1 <= top_k <= 5:
        raise ValueError("RAG_TOP_K deve ser um inteiro entre 1 e 5")
    return top_k
