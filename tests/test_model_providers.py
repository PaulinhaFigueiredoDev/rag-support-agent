from types import SimpleNamespace

from emporio_assistant import config
from emporio_assistant.providers import ollama_provider


def test_local_model_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "modelo-local")
    assert config.get_chat_model() == "modelo-local"


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
