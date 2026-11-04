import json
from types import SimpleNamespace

import pytest

from emporio_assistant import agent
from emporio_assistant.providers import ollama_provider


def ollama_message(content=None, calls=None):
    calls = calls or []
    message = SimpleNamespace(content=content, tool_calls=calls)
    message.model_dump = lambda **kwargs: {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in calls
        ],
    }
    return message


def ollama_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def return_policy_evidence():
    return [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). O cliente pode solicitar "
            "a devolução em até 7 (sete) dias corridos após o recebimento do produto. "
            "O produto deve estar em sua embalagem original, sem sinais de uso, com "
            "todos os acessórios e manuais. O reembolso será realizado na mesma "
            "forma de pagamento original em até 10 dias úteis. O frete de devolução "
            "é por conta da loja em caso de arrependimento."
        ),
        "score": 0.9,
    }]


def test_registry_and_argument_validation_reject_untrusted_calls():
    assert set(agent.TOOL_REGISTRY) == {
        "search_products", "get_order", "get_product_price", "consult_policy"
    }
    assert agent.execute_tool("delete_order", {}) == {"error": "Ferramenta não permitida"}
    assert agent.execute_tool("search_products", {"limit": True}) == {"error": "Argumentos inválidos"}
    assert agent.execute_tool("search_products", {"max_price": float("nan")}) == {
        "error": "Argumentos inválidos"
    }
    assert agent.execute_tool("get_order", {"order_id": 1, "email": 5}) == {
        "error": "Argumentos inválidos"
    }
    assert agent.execute_tool("search_products", {"unexpected": 1}) == {
        "error": "Argumentos inválidos"
    }


def test_ollama_product_call_returns_csv_data_and_preserves_history(monkeypatch):
    replies = [
        ollama_message(calls=[ollama_call("search_products", {"limit": 1})]),
        ollama_message(content="Encontrei um produto."),
    ]
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=replies.pop(0))

    monkeypatch.setattr(ollama_provider, "chat", fake_chat)
    messages = [{"role": "system", "content": "instruções"}]

    assert agent.run_ollama_turn(messages, "Mostre um produto") == "Encontrei um produto."
    assert [item["role"] for item in messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    result = json.loads(messages[3]["content"])
    assert len(result) == 1
    assert set(result[0]) == {"product_id", "name", "price_brl", "stock_quantity"}
    assert calls[1]["messages"][3]["tool_name"] == "search_products"


def test_ollama_stops_before_executing_calls_over_budget(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=ollama_message(
        calls=[ollama_call("search_products", {}) for _ in range(agent.MAX_TOOL_CALLS + 1)]
    )))
    messages = []

    assert agent.run_ollama_turn(messages, "Produtos") == agent.FALLBACK
    assert [item["role"] for item in messages] == ["user", "assistant"]


def test_ollama_generic_product_request_recovers_when_model_skips_tool(monkeypatch):
    replies = [
        ollama_message(content="Preciso do ID."),
        ollama_message(content="Encontrei produtos disponíveis."),
    ]
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=replies.pop(0)))
    messages = []

    assert agent.run_ollama_turn(messages, "Mostre um produto disponível com preço") == (
        "Encontrei produtos disponíveis."
    )
    assert [item["role"] for item in messages] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(messages[2]["content"])
    assert not agent._is_generic_product_request("Mostre produtos até R$ 100")
    assert not agent._is_generic_product_request("Mostre produtos de cordas")


def test_ollama_policy_request_recovers_when_model_skips_rag(monkeypatch):
    replies = [
        ollama_message(content="Informe o pedido."),
        ollama_message(content="A política está na página 4."),
    ]
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=replies.pop(0)))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [
        {"page": 4, "text": "Devolução em 7 dias", "score": 0.9}
    ])
    messages = []

    assert agent.run_ollama_turn(messages, "Qual a política de troca?") == (
        "A política está na página 4."
    )
    assert messages[2]["tool_name"] == "consult_policy"
    assert json.loads(messages[2]["content"])[0]["page"] == 4


def test_ollama_policy_request_fails_closed_if_index_is_unavailable(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="A política permite tudo.")
    ))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: (_ for _ in ()).throw(
        FileNotFoundError("índice ausente")
    ))
    messages = []

    assert agent.run_ollama_turn(messages, "Qual a política de troca?") == agent.FALLBACK
    assert [item["role"] for item in messages] == ["user", "assistant"]


def test_return_dialogue_asks_purchase_channel_then_uses_session_history(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("A resposta deve ser orientada pelo fluxo local")
    ))
    queries = []

    def fake_policy(question):
        queries.append(question)
        return return_policy_evidence()

    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", fake_policy)
    messages = [{"role": "system", "content": "instruções"}]
    question = "Preciso devolver um violão; qual a política?"

    assert agent.run_ollama_turn(messages, question) == agent.PURCHASE_QUESTION
    assert queries == []
    answer = agent.run_ollama_turn(messages, "Foi online")

    assert queries == [question]
    assert "7 dias corridos" in answer
    assert "Como a compra foi online" in answer
    assert [item["role"] for item in messages] == [
        "system", "user", "assistant", "user", "assistant", "tool", "assistant"
    ]
    assert json.loads(messages[5]["content"])[0]["page"] == 4


def test_return_dialogue_handles_in_store_followup_without_applying_online_rule(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("A resposta deve ser orientada pelo fluxo local")
    ))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: return_policy_evidence())
    messages = []

    assert agent.run_ollama_turn(messages, "Quero devolver um violão") == agent.PURCHASE_QUESTION
    assert agent.run_ollama_turn(messages, "Não sei") == agent.PURCHASE_QUESTION
    answer = agent.run_ollama_turn(messages, "Comprei presencialmente")

    assert "não estabelece nessa seção" in answer
    assert "compras presenciais" in answer
    assert "7 dias" not in answer


def test_policy_answer_keeps_online_condition_and_page_from_retrieved_text():
    evidence = [{
        "page": 4,
        "text": "4.1 Direito de Arrependimento (Compras Online). Devolução em 7 dias.",
        "score": 0.9,
    }]
    answer = agent.finalize_policy_answer("A devolução pode ser solicitada em 7 dias.", evidence)

    assert "compras online" in answer
    assert "página 4" in answer
    assert agent.finalize_policy_answer("Para compras online: 7 dias (p. 4).", evidence) == (
        "Para compras online: 7 dias (p. 4)."
    )


def test_return_policy_answer_excludes_preference_exchange_claims():
    evidence = [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). O cliente pode solicitar "
            "a devolução em até 7 (sete) dias corridos após o recebimento do produto. "
            "O produto deve estar em sua embalagem original, sem sinais de uso, com "
            "todos os acessórios e manuais. O reembolso será realizado na mesma "
            "forma de pagamento original em até 10 dias úteis. O frete de devolução "
            "é por conta da loja em caso de arrependimento. 4.3 Troca por Preferência: "
            "mediante disponibilidade; diferenças de valor serão cobradas."
        ),
        "score": 0.9,
    }]

    answer = agent.finalize_policy_answer(
        "Devolução depende da disponibilidade; diferenças de valor serão cobradas.",
        evidence,
        "Preciso devolver um violão, qual a política de devolução?",
    )

    assert "compras online" in answer or "compra do produto foi online" in answer
    assert "7 dias corridos após o recebimento" in answer
    assert "página 4, seção 4.1" in answer
    assert "disponibilidade" not in answer
    assert "diferenças de valor" not in answer
    assert agent.finalize_policy_answer("resposta", evidence[:0], "Quero devolver") == "resposta"
