import os

from dotenv import load_dotenv

load_dotenv()


def get_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
