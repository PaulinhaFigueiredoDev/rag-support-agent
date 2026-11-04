"""Ferramenta de consulta aos trechos de políticas da loja."""

from emporio_assistant.rag.retriever import retrieve_policy


def consult_policy(question: str) -> list[dict]:
    return retrieve_policy(question, top_k=3)
