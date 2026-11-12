"""Conduz a pergunta de acompanhamento sobre devolução."""

import re
from dataclasses import dataclass

from loja_exemplo_assistant.policy import (
    in_store_return_summary,
    normalize_text,
    online_return_summary,
)
from loja_exemplo_assistant.prompts import FALLBACK, PURCHASE_QUESTION
from loja_exemplo_assistant.session import Session
from loja_exemplo_assistant.tools.registry import execute_tool


@dataclass(frozen=True)
class ReturnResolution:
    answer: str
    evidence: list[dict] | None = None
    lookup_question: str | None = None


def is_return_intent(question: str) -> bool:
    normalized = normalize_text(question)
    return bool(re.search(r"\b(devolver|devolucao|arrependimento)\b", normalized)) and not bool(
        re.search(r"\b(defeito|troca|garantia)\b", normalized)
    )


def _purchase_channel(text: str) -> str | None:
    normalized = normalize_text(text)
    negated_online = any(phrase in normalized for phrase in (
        "nao foi online", "nao for online", "nao e online", "nao comprei online",
    ))
    in_store = "presencial" in normalized or "loja fisica" in normalized or negated_online
    online = ("online" in normalized and not negated_online) or "pela internet" in normalized
    if in_store and online:
        return None
    if in_store:
        return "presencial"
    if online:
        return "online"
    return None


def session_from_messages(messages: list[dict]) -> Session:
    """Recupera a pendência em chamadas antigas que só guardam mensagens."""
    pending = None
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == PURCHASE_QUESTION:
        pending = next(
            (item["content"] for item in reversed(messages[:-1])
             if item.get("role") == "user" and is_return_intent(item.get("content", ""))),
            None,
        )
    return Session(messages=messages, pending_return_question=pending)


def resolve_return(session: Session, user_text: str) -> ReturnResolution | None:
    """Consulta a política após confirmar o canal de compra."""
    previous_question = session.pending_return_question
    if previous_question is None and not is_return_intent(user_text):
        return None

    channel = _purchase_channel(user_text)
    if channel is None:
        session.pending_return_question = previous_question or user_text
        return ReturnResolution(PURCHASE_QUESTION)

    session.pending_return_question = None
    lookup_question = previous_question or user_text
    evidence = execute_tool("consult_policy", {"question": lookup_question})
    if not isinstance(evidence, list) or not evidence:
        return ReturnResolution(FALLBACK)
    if channel == "online":
        answer = online_return_summary(evidence, confirmed_online=True) or FALLBACK
    else:
        answer = in_store_return_summary(evidence) or FALLBACK
    return ReturnResolution(answer, evidence, lookup_question)
