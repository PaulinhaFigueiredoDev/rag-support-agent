# Testes da orquestração do agente, das chamadas de ferramentas e das respostas com dados.

import json
from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from loja_exemplo_assistant import agent
from loja_exemplo_assistant.providers import ollama_provider


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
            "a devolução em até 7 (sete) dias corridos após o recebimento do produto, "
            "sem necessidade de justificativa. "
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
        ollama_message(content="O produto custa R$ 1,00.", calls=[
            ollama_call("search_products", {"limit": 1})
        ]),
    ]
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=replies.pop(0))

    monkeypatch.setattr(ollama_provider, "chat", fake_chat)
    messages = [{"role": "system", "content": "instruções"}]

    answer = agent.run_ollama_turn(messages, "Mostre um produto")
    assert "Produtos disponíveis:" in answer
    assert "R$ 1,00" not in answer
    assert [item["role"] for item in messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    result = json.loads(messages[3]["content"])
    assert len(result) == 1
    assert set(result[0]) == {"product_id", "name", "price_brl", "stock_quantity"}
    assert len(calls) == 1
    assert str(result[0]["name"]) in answer


def test_ollama_stops_before_executing_calls_over_budget(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=ollama_message(
        calls=[ollama_call("search_products", {}) for _ in range(agent.MAX_TOOL_CALLS + 1)]
    )))
    messages = []

    assert agent.run_ollama_turn(messages, "Produtos") == agent.FALLBACK
    assert [item["role"] for item in messages] == ["user", "assistant"]


@pytest.mark.parametrize("failure", [
    ConnectionError("Ollama indisponível"),
    httpx.ReadTimeout("Tempo esgotado"),
    ResponseError("Modelo não encontrado", status_code=404),
])
def test_ollama_failures_return_useful_answer_and_keep_history(monkeypatch, failure):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: (_ for _ in ()).throw(failure))
    messages = []

    answer = agent.run_ollama_turn(messages, "Olá")

    assert answer == agent.MODEL_UNAVAILABLE
    assert "Ollama" in answer
    assert messages == [
        {"role": "user", "content": "Olá"},
        {"role": "assistant", "content": answer},
    ]


def test_malformed_ollama_response_returns_useful_answer(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=None))

    assert agent.run_ollama_turn([], "Olá") == agent.MODEL_UNAVAILABLE


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("Embedding demorou demais"),
    ResponseError("Modelo de embedding ausente", status_code=404),
])
def test_policy_embedding_failure_does_not_interrupt_return_dialogue(monkeypatch, failure):
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: (
        _ for _ in ()
    ).throw(failure))
    messages = []

    assert agent.run_ollama_turn(messages, "Quero devolver uma compra online") == agent.FALLBACK
    assert [item["role"] for item in messages] == ["user", "assistant"]


def test_ollama_generic_product_request_recovers_when_model_skips_tool(monkeypatch):
    replies = [ollama_message(content="Preciso do ID.")]
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=replies.pop(0)))
    messages = []

    answer = agent.run_ollama_turn(messages, "Mostre um produto disponível com preço")
    assert "Produtos disponíveis:" in answer
    assert "R$" in answer
    assert "Preciso do ID" not in answer
    assert [item["role"] for item in messages] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(messages[2]["content"])


def test_price_request_uses_csv_even_when_model_invents_value(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="O produto custa R$ 1,00.")
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "Qual o preço do produto 81?")

    assert "R$ 599,90" in answer
    assert "R$ 1,00" not in answer
    assert messages[2]["tool_name"] == "get_product_price"


def test_price_request_asks_for_id_and_keeps_pix_context(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="O preço é R$ 1,00.")
    ))
    messages = []

    assert agent.run_ollama_turn(messages, "Quanto custa no Pix?") == (
        "Para consultar o preço, informe o ID do produto."
    )
    answer = agent.run_ollama_turn(messages, "81")

    assert "desconto de 5%" in answer
    assert "R$ 569,91" in answer


def test_order_request_requires_customer_email_and_uses_csv(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="O pedido foi cancelado.")
    ))
    messages = []

    assert agent.run_ollama_turn(messages, "Qual o status do pedido 1?") == (
        "Para consultar o pedido, informe o número e o e-mail usado na compra."
    )
    assert all(item["role"] != "tool" for item in messages)

    answer = agent.run_ollama_turn(messages, "pedro.oliveira@jmail.com")

    assert "Pedido 1: status entregue" in answer
    assert "BRAB1234567BR" in answer
    assert "cancelado" not in answer
    assert "pedro.oliveira@jmail.com" not in answer


def test_model_cannot_query_order_with_invented_email(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(calls=[ollama_call("get_order", {
            "order_id": 1, "email": "pedro.oliveira@jmail.com",
        })])
    ))
    messages = []

    assert agent.run_ollama_turn(messages, "Qual o status do pedido 1?") == (
        "Para consultar o pedido, informe o número e o e-mail usado na compra."
    )
    assert all(item["role"] != "tool" for item in messages)


def test_stock_request_for_id_uses_matching_product_only(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Há 900 unidades disponíveis.")
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "O produto 81 está disponível em estoque?")

    assert "Yamaha C40" in answer
    assert "12 em estoque" in answer
    assert "900" not in answer
    result = json.loads(messages[2]["content"])
    assert [item["product_id"] for item in result] == [81]


def test_stock_followup_uses_last_quoted_product(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Não há estoque.")
    ))
    messages = []

    assert "Produto 81" in agent.run_ollama_turn(messages, "Qual o preço do produto 81 no Pix?")
    messages[-1]["content"] = "Preço consultado com sucesso."
    answer = agent.run_ollama_turn(messages, "esta disponivel em estoque?")

    assert "Yamaha C40" in answer
    assert "12 em estoque" in answer
    assert json.loads(messages[-2]["content"])[0]["product_id"] == 81


def test_stock_followup_without_clear_product_asks_for_id(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Há estoque.")
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "esta disponivel em estoque?")

    assert answer == "De qual produto você quer consultar o estoque? Informe o ID."
    assert all(item["role"] != "tool" for item in messages)
    answer = agent.run_ollama_turn(messages, "81")
    assert "12 em estoque" in answer


def test_stock_followup_after_multiple_products_asks_for_id(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Há estoque.")
    ))
    messages = []

    assert "Produtos disponíveis:" in agent.run_ollama_turn(messages, "Mostre produtos")
    answer = agent.run_ollama_turn(messages, "Algum está disponível em estoque?")

    assert answer == "De qual produto você quer consultar o estoque? Informe o ID."


def test_stock_request_corrects_model_choice_of_price_tool(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(calls=[ollama_call("get_product_price", {"product_id": 81})])
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "O produto 81 está disponível em estoque?")

    assert "12 em estoque" in answer
    assert messages[2]["tool_name"] == "search_products"


def test_product_search_honors_category_and_price_limit(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Todos os violões custam R$ 1,00.")
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "Mostre violões até R$ 500")

    result = json.loads(messages[2]["content"])
    assert result
    assert all(item["price_brl"] <= 500 for item in result)
    assert all(item["name"] in answer for item in result)
    assert "R$ 1,00" not in answer


def test_model_product_id_cannot_replace_customer_requested_id(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(calls=[ollama_call("get_product_price", {"product_id": 82})])
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "Qual o preço do produto 81?")

    assert "Produto 81" in answer
    assert "R$ 599,90" in answer
    assert json.loads(messages[2]["content"])["original_price"] == "599.9"


def test_pix_discount_does_not_leak_to_next_product(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Preço inventado")
    ))
    messages = []

    assert "desconto de 5%" in agent.run_ollama_turn(messages, "Preço do produto 81 no Pix?")
    answer = agent.run_ollama_turn(messages, "Qual o preço do produto 82?")

    assert "Produto 82" in answer
    assert "desconto de 0%" in answer


def test_new_topic_replaces_pending_order_question(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Resposta inventada")
    ))
    messages = []

    assert "informe o número" in agent.run_ollama_turn(messages, "Qual o status do pedido?")
    answer = agent.run_ollama_turn(messages, "Mostre um produto")

    assert "Produtos disponíveis:" in answer
    assert "pedido" not in answer.casefold()


def test_unavailable_product_data_does_not_return_model_claim(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="O produto custa R$ 1,00.")
    ))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "get_product_price", lambda product_id, payment=None: (
        _ for _ in ()
    ).throw(FileNotFoundError("CSV ausente")))

    assert agent.run_ollama_turn([], "Qual o preço do produto 81?") == agent.FALLBACK


def test_delivery_policy_uses_current_question_without_prior_product_context(monkeypatch):
    replies = [
        ollama_message(content="R$ 1,00"),
        ollama_message(content="Não sei o prazo para o produto 81."),
        ollama_message(content="Na região metropolitana, o prazo é de 1 a 3 dias úteis."),
    ]
    requests = []

    def fake_chat(**kwargs):
        requests.append([item.copy() for item in kwargs["messages"]])
        return SimpleNamespace(message=replies.pop(0))

    monkeypatch.setattr(ollama_provider, "chat", fake_chat)
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [
        {"page": 4, "text": (
            "5.1 Entregas na Região Metropolitana de Campo Grande. "
            "Prazo de entrega: 1 a 3 dias úteis."
        )}
    ])
    messages = [{"role": "system", "content": "instruções"}]

    assert "Produto 81" in agent.run_ollama_turn(messages, "Qual o preço do produto 81 no Pix?")
    answer = agent.run_ollama_turn(messages, "qual prazo para entrega?")

    assert "1 a 3 dias úteis" in answer
    assert "página 4" in answer
    assert "produto 81" not in answer.casefold()
    assert [item["role"] for item in requests[1]] == ["system", "user"]
    assert [item["role"] for item in requests[2]] == ["system", "user", "assistant", "tool"]


def test_order_delivery_status_still_uses_order_lookup(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="Entrega em 3 dias.")
    ))
    messages = []

    answer = agent.run_ollama_turn(messages, "Qual o status de entrega do pedido 1 via SEDEX?")

    assert "informe o número e o e-mail" in answer
    assert all(item["role"] != "tool" for item in messages)


def test_sedex_followup_uses_shipping_policy_without_model(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("A modalidade deve usar o manual, sem nova resposta do modelo")
    ))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [
        {"page": 5, "text": (
            "Envios para fora da região metropolitana de Campo Grande. "
            "PAC (Correios) 5 a 12 dias úteis. SEDEX (Correios) 2 a 5 dias úteis. "
            "Jadlog 3 a 8 dias úteis."
        )}
    ])
    messages = []

    answer = agent.run_ollama_turn(messages, "quero sedex")

    assert "SEDEX: 2 a 5 dias úteis" in answer
    assert "página 5" in answer
    assert "região metropolitana de Campo Grande (Manual de Políticas, página 4" not in answer
    assert messages[2]["tool_name"] == "consult_policy"


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


def test_policy_request_without_relevant_evidence_reports_not_found(monkeypatch):
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(
        message=ollama_message(content="A loja aceita animais.")
    ))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [])

    answer = agent.run_ollama_turn([], "Qual a política para animais na loja?")

    assert answer == agent.POLICY_NOT_FOUND


def test_model_cannot_answer_after_empty_policy_tool_result(monkeypatch):
    replies = [
        ollama_message(calls=[ollama_call("consult_policy", {"question": "Política para animais"})]),
        ollama_message(content="A loja aceita animais."),
    ]
    monkeypatch.setattr(ollama_provider, "chat", lambda **kwargs: SimpleNamespace(message=replies.pop(0)))
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [])

    assert agent.run_ollama_turn([], "Vocês aceitam animais?") == agent.POLICY_NOT_FOUND


def test_return_dialogue_reports_when_policy_evidence_is_missing(monkeypatch):
    monkeypatch.setitem(agent.TOOL_REGISTRY, "consult_policy", lambda question: [])

    assert agent.run_ollama_turn([], "Quero devolver uma compra online") == agent.POLICY_NOT_FOUND


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
            "a devolução em até 7 (sete) dias corridos após o recebimento do produto, "
            "sem necessidade de justificativa. "
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
