"""Comunicação e formato de mensagens do Ollama."""

import json
from dataclasses import dataclass

from httpx import RequestError as HttpRequestError
from ollama import Client, RequestError, ResponseError

from loja_exemplo_assistant.config import get_chat_model

chat = Client(timeout=120).chat


class ModelUnavailableError(Exception):
    """O modelo local não respondeu de forma utilizável."""


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
    try:
        response = chat(
            model=get_chat_model(),
            messages=messages,
            tools=schemas,
            options={"num_ctx": 4096, "temperature": 0},
        )
        message = response.message
        calls = [
            ToolCall(call.function.name, call.function.arguments)
            for call in (message.tool_calls or [])
        ]
        if not isinstance(message.content, (str, type(None))) or not all(
            isinstance(call.name, str) and call.name and isinstance(call.arguments, (dict, str))
            for call in calls
        ):
            raise ValueError("Resposta do modelo inválida")
        assistant_message = message.model_dump(exclude_none=True)
        if not isinstance(assistant_message, dict):
            raise ValueError("Mensagem do modelo inválida")
        return ModelResponse(message.content, calls, assistant_message)
    except (ConnectionError, HttpRequestError, RequestError, ResponseError,
            AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError from exc


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
    from loja_exemplo_assistant.agent import run_turn
    from loja_exemplo_assistant.return_dialogue import session_from_messages

    return run_turn(session_from_messages(messages), user_text)
