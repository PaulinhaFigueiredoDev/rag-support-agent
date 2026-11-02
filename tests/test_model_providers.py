from types import SimpleNamespace

import pytest

from emporio_assistant import config
from emporio_assistant.providers import groq_provider, ollama_provider


def test_provider_configuration(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", " Groq ")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "modelo-remoto")
    assert config.get_provider() == "groq"
    assert config.get_chat_model() == "modelo-remoto"

    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "modelo-local")
    assert config.get_chat_model() == "modelo-local"

    monkeypatch.setenv("AI_PROVIDER", "invalido")
    with pytest.raises(ValueError, match="AI_PROVIDER"):
        config.get_provider()


def test_ollama_chat_receives_prompt_and_local_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
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


def test_groq_requires_key_before_creating_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def unexpected_client(**kwargs):
        raise AssertionError("Cliente remoto não deve ser criado sem chave")

    monkeypatch.setattr(groq_provider, "OpenAI", unexpected_client)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        groq_provider.ask_groq("Olá")


def test_groq_chat_uses_selected_model_and_endpoint(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "modelo-remoto")
    monkeypatch.setenv("GROQ_API_KEY", "chave-ficticia")
    calls = []
    client_options = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="resposta remota"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    def fake_client(**kwargs):
        client_options.append(kwargs)
        return client

    monkeypatch.setattr(groq_provider, "OpenAI", fake_client)

    assert groq_provider.ask_groq("Olá") == "resposta remota"
    assert client_options == [{
        "api_key": "chave-ficticia",
        "base_url": "https://api.groq.com/openai/v1",
    }]
    assert calls == [{
        "model": "modelo-remoto",
        "messages": [{"role": "user", "content": "Olá"}],
        "temperature": 0,
    }]
