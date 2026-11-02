from ollama import chat

from emporio_assistant.config import get_chat_model


def ask_ollama(prompt: str) -> str:
    response = chat(
        model=get_chat_model(),
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048, "temperature": 0},
    )
    return response.message.content or ""
