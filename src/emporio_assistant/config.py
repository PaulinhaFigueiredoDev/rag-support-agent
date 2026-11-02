import os

from dotenv import load_dotenv

load_dotenv()


def get_provider() -> str:
    provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "groq"}:
        raise ValueError("AI_PROVIDER deve ser ollama ou groq")
    return provider


def get_chat_model() -> str:
    if get_provider() == "ollama":
        return os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
    return os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
