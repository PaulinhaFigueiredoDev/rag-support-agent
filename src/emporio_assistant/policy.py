"""Regras de resposta baseadas nas políticas recuperadas."""

import re
import unicodedata
from dataclasses import dataclass

from emporio_assistant.session import Session
from emporio_assistant.tools.registry import execute_tool

FALLBACK = "Não consegui concluir a consulta. Posso ajudar com outra pergunta?"
PURCHASE_QUESTION = "A compra foi online ou presencial?"


@dataclass(frozen=True)
class ReturnResolution:
    answer: str
    evidence: list[dict] | None = None
    lookup_question: str | None = None


def is_policy_request(text: str) -> bool:
    normalized = _normalize(text)
    return bool(re.search(r"\b(politica|troca|devolucao|arrependimento|garantia)\b", normalized))


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _simple_return_request(question: str) -> bool:
    normalized = _normalize(question)
    return bool(re.search(r"\b(devolver|devolucao|arrependimento)\b", normalized)) and not bool(
        re.search(r"\b(defeito|troca|garantia|presencial)\b|loja fisica", normalized)
    )


def is_return_intent(question: str) -> bool:
    normalized = _normalize(question)
    return bool(re.search(r"\b(devolver|devolucao|arrependimento)\b", normalized)) and not bool(
        re.search(r"\b(defeito|troca|garantia)\b", normalized)
    )


def _purchase_channel(text: str) -> str | None:
    normalized = _normalize(text)
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


def _policy_page(evidence: list[dict]) -> int | None:
    heading = "4.1 direito de arrependimento (compras online)"
    return next(
        (item["page"] for item in evidence if heading in _normalize(" ".join(item["text"].split()))),
        None,
    )


def _return_policy_summary(evidence: list[dict], confirmed_online: bool = False) -> str | None:
    relevant_page = _policy_page(evidence)
    if relevant_page is None:
        return None
    source = _normalize(" ".join(
        " ".join(item["text"].split()) for item in evidence if item["page"] == relevant_page
    ))
    required_facts = (
        "7 (sete) dias corridos apos o recebimento",
        "embalagem original, sem sinais de uso",
        "acessorios e manuais",
        "mesma forma de pagamento original em ate 10 dias uteis",
        "frete de devolucao e por conta da loja em caso de arrependimento",
    )
    if not all(fact in source for fact in required_facts):
        return None
    opening = "Como a compra foi online, " if confirmed_online else "Se a compra do produto foi online, "
    return (
        opening + "você pode solicitar a devolução por "
        "arrependimento em até 7 dias corridos após o recebimento, sem justificativa. "
        "O produto deve estar na embalagem original, sem sinais de uso, com todos "
        "os acessórios e manuais. O reembolso ocorre na mesma forma de pagamento "
        "original em até 10 dias úteis. Em caso de arrependimento, o frete de "
        "devolução é por conta da loja.\n\n"
        "Para iniciar a solicitação, entre em contato com o atendimento humano. "
        "Aqui posso apenas consultar o status de um pedido.\n\n"
        f"Fonte: Manual de Políticas, página {relevant_page}, seção 4.1."
    )


def _in_store_return_summary(evidence: list[dict]) -> str | None:
    page = _policy_page(evidence)
    if page is None:
        return None
    return (
        "A seção 4.1 do manual descreve a devolução por arrependimento para "
        "compras online. Ela não estabelece nessa seção uma regra de devolução "
        "por arrependimento para compras presenciais. Para verificar as opções "
        "no seu caso, entre em contato com o atendimento humano.\n\n"
        f"Fonte: Manual de Políticas, página {page}, seção 4.1."
    )


def finalize_policy_answer(answer: str, evidence: list[dict], question: str = "") -> str:
    if not evidence or answer == FALLBACK:
        return answer
    if _simple_return_request(question):
        return _return_policy_summary(evidence) or FALLBACK
    source_text = _normalize(" ".join(item["text"] for item in evidence))
    normalized_answer = _normalize(answer)
    if (
        "direito de arrependimento (compras online)" in source_text
        and re.search(r"\b7\s*(?:\([^)]*\)\s*)?dias?\b", normalized_answer)
        and "online" not in normalized_answer
    ):
        answer += "\n\nO prazo de arrependimento citado se aplica a compras online."
    if not re.search(r"\bpagina\b|\bp\.\s*\d+", normalized_answer):
        answer += f"\n\nFonte: Manual de Políticas, página {evidence[0]['page']}."
    return answer


def resolve_return(session: Session, user_text: str) -> ReturnResolution | None:
    """Resolve o diálogo de devolução sem depender de um modelo."""
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
        answer = _return_policy_summary(evidence, confirmed_online=True) or FALLBACK
    else:
        answer = _in_store_return_summary(evidence) or FALLBACK
    return ReturnResolution(answer, evidence, lookup_question)
