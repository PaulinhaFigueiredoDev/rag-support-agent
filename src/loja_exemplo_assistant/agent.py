"""Orquestra um turno de atendimento com o modelo local."""

from loja_exemplo_assistant.policy import (
    finalize_policy_answer,
    is_policy_request,
)
from loja_exemplo_assistant.prompts import FALLBACK, PURCHASE_QUESTION
from loja_exemplo_assistant.providers import ollama_provider
from loja_exemplo_assistant.providers.ollama_provider import ToolCall
from loja_exemplo_assistant.return_dialogue import resolve_return, session_from_messages
from loja_exemplo_assistant.session import Session
from loja_exemplo_assistant.tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS, execute_tool

MAX_TOOL_CALLS = 4


def _is_generic_product_request(text: str) -> bool:
    question = text.casefold().strip()
    return (
        question.startswith(("mostre", "liste", "quais", "sugira"))
        and ("produto" in question or "instrumento" in question)
        and not any(character.isdigit() for character in question)
        and not any(phrase in question for phrase in (" até ", " abaixo ", " de ", " categoria "))
    )


def _required_tool(user_text: str, used_tools: set[str]) -> tuple[str, dict] | None:
    if is_policy_request(user_text) and "consult_policy" not in used_tools:
        return "consult_policy", {"question": user_text}
    if _is_generic_product_request(user_text) and "search_products" not in used_tools:
        return "search_products", {"limit": 5}
    return None


def _successful_result(name: str, result: object) -> bool:
    return not (isinstance(result, dict) and "error" in result) and (name != "consult_policy" or bool(result))


def run_turn(session: Session, user_text: str) -> str:
    """Conduz uma interação e atualiza o histórico da sessão."""
    local = resolve_return(session, user_text)
    if local is not None:
        session.add_user(user_text)
        if local.evidence and local.lookup_question:
            arguments = {"question": local.lookup_question}
            call = ToolCall("consult_policy", arguments)
            session.messages.append(ollama_provider.forced_call_message(call))
            session.messages.append(ollama_provider.tool_result_message(call, local.evidence))
        session.add_assistant(local.answer)
        return local.answer

    session.add_user(user_text)
    calls_used = 0
    used_tools: set[str] = set()
    policy_evidence: list[dict] = []

    while True:
        response = ollama_provider.complete(session.messages, TOOL_SCHEMAS)
        if not response.calls:
            required = _required_tool(user_text, used_tools)
            if required and calls_used < MAX_TOOL_CALLS:
                name, arguments = required
                result = execute_tool(name, arguments)
                if not _successful_result(name, result):
                    session.add_assistant(FALLBACK)
                    return FALLBACK
                call = ToolCall(name, arguments)
                session.messages.append(ollama_provider.forced_call_message(call))
                session.messages.append(ollama_provider.tool_result_message(call, result))
                if name == "consult_policy":
                    policy_evidence = result
                used_tools.add(name)
                calls_used += 1
                continue

            answer = response.content or FALLBACK
            if is_policy_request(user_text) and not policy_evidence:
                answer = FALLBACK
            answer = finalize_policy_answer(answer, policy_evidence, user_text)
            session.add_assistant(answer)
            return answer

        if calls_used + len(response.calls) > MAX_TOOL_CALLS:
            session.add_assistant(FALLBACK)
            return FALLBACK

        session.messages.append(response.assistant_message)
        for call in response.calls:
            result = execute_tool(call.name, call.arguments)
            if _successful_result(call.name, result):
                used_tools.add(call.name)
                if call.name == "consult_policy":
                    policy_evidence = result
            session.messages.append(ollama_provider.tool_result_message(call, result))
        calls_used += len(response.calls)


def run_ollama_turn(messages: list[dict], user_text: str) -> str:
    """Compatibilidade com a API anterior baseada em uma lista de mensagens."""
    return run_turn(session_from_messages(messages), user_text)
