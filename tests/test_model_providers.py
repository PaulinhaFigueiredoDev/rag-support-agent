from types import SimpleNamespace

import pytest

from loja_exemplo_assistant import config
from loja_exemplo_assistant.providers import ollama_provider
from loja_exemplo_assistant.tools import policies


def test_local_model_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "modelo-local")
    assert config.get_chat_model() == "modelo-local"


def test_rag_top_k_configuration_reaches_policy_retriever(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "4")
    calls = []
    monkeypatch.setattr(policies, "retrieve_policy", lambda question, top_k: calls.append(
        (question, top_k)
    ) or [])

    assert policies.consult_policy("Qual a política de troca?") == []
    assert calls == [("Qual a política de troca?", 4)]


def test_rag_top_k_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    assert config.get_rag_top_k() == 3


@pytest.mark.parametrize("value", ["0", "6", "texto", ""])
def test_rag_top_k_rejects_invalid_configuration(monkeypatch, value):
    monkeypatch.setenv("RAG_TOP_K", value)
    with pytest.raises(ValueError, match="RAG_TOP_K deve ser um inteiro entre 1 e 5"):
        config.get_rag_top_k()


def test_ollama_chat_receives_prompt_and_local_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="resposta local"))

    monkeypatch.setattr(ollama_provider, "chat", fake_chat)

    assert ollama_provider.ask_ollama("Olá") == "resposta local"
    assert calls == [{
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": "Olá"}],
        "options": {"num_ctx": 2048, "temperature": 0},
    }]
