import builtins

import main
from emporio_assistant.providers import ollama_provider
from emporio_assistant.tools.registry import TOOL_REGISTRY


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
    assert "Empório Assist (Ollama)" in output
    assert "Atendimento encerrado." in output


def test_main_handles_eof(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda prompt: (_ for _ in ()).throw(EOFError))
    main.main()
    assert "Atendimento encerrado." in capsys.readouterr().out


def test_main_completes_return_dialogue_with_session(monkeypatch, capsys):
    questions = iter(["Quero devolver um violão", "Foi online", "sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(questions))
    monkeypatch.setitem(TOOL_REGISTRY, "consult_policy", lambda question: [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). Devolução em até "
            "7 (sete) dias corridos após o recebimento. Embalagem original, sem sinais de uso, "
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
