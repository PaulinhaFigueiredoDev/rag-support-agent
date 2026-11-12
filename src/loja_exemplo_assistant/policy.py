"""Regras de resposta baseadas nas políticas recuperadas."""

import re
import unicodedata

from loja_exemplo_assistant.prompts import FALLBACK


def is_policy_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(re.search(r"\b(politica|troca|devolucao|arrependimento|garantia)\b", normalized))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _simple_return_request(question: str) -> bool:
    normalized = normalize_text(question)
    return bool(re.search(r"\b(devolver|devolucao|arrependimento)\b", normalized)) and not bool(
        re.search(r"\b(defeito|troca|garantia|presencial)\b|loja fisica", normalized)
    )


def _policy_page(evidence: list[dict]) -> int | None:
    return next(
        (
            item["page"] for item in evidence
            if re.search(
                r"\b4\.1\s+direito de arrependimento\s*\(\s*compras online\s*\)",
                normalize_text(item["text"]),
            )
        ),
        None,
    )


def _has_required_return_facts(source: str) -> bool:
    """Confere os fatos usados no resumo, aceitando pequenas variações de redação."""
    patterns = (
        r"\b7\s*(?:\(\s*sete\s*\)\s*)?dias? corridos?\b.{0,70}"
        r"(?:apos o|a partir do) recebimento\b",
        r"\b(?:sem (?:necessidade de )?justificativa|nao precisa de justificativa)\b",
        r"\bembalagem original\b.{0,70}\bsem sinais de uso\b",
        r"\bacessorios\b.{0,40}\bmanuais\b",
        r"\bmesma forma de pagamento original\b.{0,90}\b10 dias uteis\b",
        r"\bfrete de devolucao\b.{0,90}\b(?:por conta da loja|pago pela loja)\b",
    )
    return all(re.search(pattern, source) for pattern in patterns)


def online_return_summary(evidence: list[dict], confirmed_online: bool = False) -> str | None:
    relevant_page = _policy_page(evidence)
    if relevant_page is None:
        return None
    source = normalize_text(" ".join(
        " ".join(item["text"].split()) for item in evidence if item["page"] == relevant_page
    ))
    if not _has_required_return_facts(source):
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


def in_store_return_summary(evidence: list[dict]) -> str | None:
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
        return online_return_summary(evidence) or FALLBACK
    source_text = normalize_text(" ".join(item["text"] for item in evidence))
    normalized_answer = normalize_text(answer)
    if (
        "direito de arrependimento (compras online)" in source_text
        and re.search(r"\b7\s*(?:\([^)]*\)\s*)?dias?\b", normalized_answer)
        and "online" not in normalized_answer
    ):
        answer += "\n\nO prazo de arrependimento citado se aplica a compras online."
    if not re.search(r"\bpagina\b|\bp\.\s*\d+", normalized_answer):
        answer += f"\n\nFonte: Manual de Políticas, página {evidence[0]['page']}."
    return answer
