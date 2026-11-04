"""Comunicação e formato de mensagens do Ollama."""

import json

from ollama import chat

from emporio_assistant.config import get_chat_model
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: object


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    calls: list[ToolCall]
    assistant_message: dict


def ask_ollama(prompt: str) -> str:
    response = chat(
        model=get_chat_model(),
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048, "temperature": 0},
    )
    return response.message.content or ""


def complete(messages: list[dict], schemas: list[dict]) -> ModelResponse:
    response = chat(
        model=get_chat_model(),
        messages=messages,
        tools=schemas,
        options={"num_ctx": 4096, "temperature": 0},
    )
    message = response.message
    return ModelResponse(
        content=message.content,
        calls=[ToolCall(call.function.name, call.function.arguments) for call in (message.tool_calls or [])],
        assistant_message=message.model_dump(exclude_none=True),
    )


def forced_call_message(call: ToolCall) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }],
    }


def tool_result_message(call: ToolCall, result: object) -> dict:
    return {
        "role": "tool",
        "tool_name": call.name,
        "content": json.dumps(result, ensure_ascii=False, default=str),
    }


def run_ollama_turn(messages: list[dict], user_text: str) -> str:
    """Compatibilidade com a API anterior baseada em uma lista de mensagens."""
    from emporio_assistant.agent import run_turn
    from emporio_assistant.session import Session

    return run_turn(Session.from_messages(messages), user_text)
