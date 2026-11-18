"""Comunicação e formato de mensagens do Ollama."""

import json
from dataclasses import dataclass

from httpx import RequestError as HttpRequestError
from ollama import Client, RequestError, ResponseError

from loja_exemplo_assistant.config import get_chat_model

chat = Client(timeout=120).chat


class ModelUnavailableError(Exception):
    """O modelo local não respondeu de forma utilizável."""


# Representa o nome de uma ferramenta e os argumentos propostos pelo modelo.
@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: object


# Agrupa o texto, as chamadas de ferramenta e a mensagem original retornada pelo modelo.
@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    calls: list[ToolCall]
    assistant_message: dict


# Envia uma solicitação simples ao modelo local e retorna o texto produzido.
def ask_ollama(prompt: str) -> str:
    response = chat(
        model=get_chat_model(),
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048, "temperature": 0},
    )
    return response.message.content or ""


# Envia histórico e schemas de ferramentas ao Ollama e normaliza a resposta ou sinaliza indisponibilidade.
def complete(messages: list[dict], schemas: list[dict]) -> ModelResponse:
    # Usa temperatura zero e permite tool calls para tornar o fluxo previsível.
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
        # Verifica a estrutura antes que o agente use conteúdo ou argumentos do modelo.
        if not isinstance(message.content, (str, type(None))) or not all(
            isinstance(call.name, str) and call.name and isinstance(call.arguments, (dict, str))
            for call in calls
        ):
            raise ValueError("Resposta do modelo inválida")
        assistant_message = message.model_dump(exclude_none=True)
        if not isinstance(assistant_message, dict):
            raise ValueError("Mensagem do modelo inválida")
        return ModelResponse(message.content, calls, assistant_message)
    # Converte exceções do cliente e respostas malformadas em um único erro do provedor.
    except (ConnectionError, HttpRequestError, RequestError, ResponseError,
            AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError from exc


def forced_call_message(call: ToolCall) -> dict:
    # Registra uma chamada montada pela aplicação no formato de mensagem do Ollama.
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }],
    }


def tool_result_message(call: ToolCall, result: object) -> dict:
    # Associa o resultado ao nome da ferramenta e o serializa em JSON para o histórico.
    return {
        "role": "tool",
        "tool_name": call.name,
        "content": json.dumps(result, ensure_ascii=False, default=str),
    }


def run_ollama_turn(messages: list[dict], user_text: str) -> str:
    """Compatibilidade com a API anterior baseada em uma lista de mensagens."""
    # Faz importações locais para evitar dependência circular com o módulo do agente.
    from loja_exemplo_assistant.agent import run_turn
    from loja_exemplo_assistant.return_dialogue import session_from_messages

    return run_turn(session_from_messages(messages), user_text)
