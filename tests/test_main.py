import builtins
from types import SimpleNamespace

import main
from loja_exemplo_assistant.providers import ollama_provider
from loja_exemplo_assistant.tools.registry import TOOL_REGISTRY


def test_main_keeps_session_history(monkeypatch, capsys):
    questions = iter(["Olá", "Outro assunto", "sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(questions))
    histories = []

    def fake_turn(session, question):
        histories.append((id(session), question, len(session.messages)))
        session.add_user(question)
        session.add_assistant("OK")
        return "OK"

    monkeypatch.setattr(main, "run_turn", fake_turn)
    main.main()

    assert histories[0][0] == histories[1][0]
    assert [item[1:] for item in histories] == [("Olá", 1), ("Outro assunto", 3)]
    output = capsys.readouterr().out
    assert "Loja Exemplo Assist (Ollama)" in output
    assert "Atendimento encerrado." in output


def test_main_handles_eof(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda prompt: (_ for _ in ()).throw(EOFError))
    main.main()
    assert "Atendimento encerrado." in capsys.readouterr().out


def test_main_continues_after_ollama_connection_failure(monkeypatch, capsys):
    questions = iter(["Olá", "Você está aí?", "sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(questions))
    histories = []

    def fake_chat(**kwargs):
        histories.append([item.copy() for item in kwargs["messages"]])
        if len(histories) == 1:
            raise ConnectionError("Ollama indisponível")
        message = SimpleNamespace(content="Sim, posso ajudar.", tool_calls=[])
        message.model_dump = lambda **kwargs: {"role": "assistant", "content": message.content}
        return SimpleNamespace(message=message)

    monkeypatch.setattr(ollama_provider, "chat", fake_chat)
    main.main()

    output = capsys.readouterr().out
    assert "Verifique se o Ollama está em execução" in output
    assert "Sim, posso ajudar." in output
    assert [item["role"] for item in histories[1]] == ["system", "user", "assistant", "user"]


def test_main_completes_return_dialogue_with_session(monkeypatch, capsys):
    questions = iter(["Quero devolver um violão", "Foi online", "sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(questions))
    monkeypatch.setitem(TOOL_REGISTRY, "consult_policy", lambda question: [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). Devolução em até "
            "7 (sete) dias corridos após o recebimento, sem necessidade de justificativa. "
            "Embalagem original, sem sinais de uso, "
            "com todos os acessórios e manuais. Reembolso na mesma forma de pagamento original "
            "em até 10 dias úteis. O frete de devolução é por conta da loja em caso de arrependimento."
        ),
    }])

    def unexpected_model(*args, **kwargs):
        raise AssertionError("O diálogo de devolução deve usar a política local")

    monkeypatch.setattr(ollama_provider, "complete", unexpected_model)
    main.main()

    output = capsys.readouterr().out
    assert "A compra foi online ou presencial?" in output
    assert "Como a compra foi online" in output
    assert "página 4" in output
