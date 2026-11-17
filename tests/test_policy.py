from loja_exemplo_assistant.policy import (
    delivery_time_summary, finalize_policy_answer, online_return_summary,
    requested_shipping_modes, shipping_mode_summary,
)


def test_return_summary_accepts_equivalent_wording_from_manual():
    evidence = [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). "
            "Devolução em até 7 dias corridos, a partir do recebimento. "
            "Não precisa de justificativa. Embalagem original, sem sinais de uso, "
            "com acessórios e manuais. Reembolso na mesma forma de pagamento "
            "original no prazo de 10 dias úteis. Frete de devolução pago pela loja "
            "em caso de arrependimento."
        ),
    }]

    answer = online_return_summary(evidence, confirmed_online=True)

    assert answer is not None
    assert "7 dias corridos após o recebimento" in answer
    assert "página 4, seção 4.1" in answer


def test_return_summary_rejects_evidence_without_shipping_rule():
    evidence = [{
        "page": 4,
        "text": (
            "4.1 Direito de Arrependimento (Compras Online). "
            "7 dias corridos após o recebimento, sem justificativa. "
            "Embalagem original, sem sinais de uso, com acessórios e manuais. "
            "Reembolso na mesma forma de pagamento original em até 10 dias úteis."
        ),
    }]

    assert online_return_summary(evidence) is None


def test_delivery_answer_uses_each_verified_shipping_mode():
    evidence = [
        {"page": 4, "text": (
            "5.1 Entregas na Região Metropolitana de Campo Grande. "
            "Prazo de entrega: 1 a 3 dias úteis."
        )},
        {"page": 5, "text": (
            "Envios para fora da região metropolitana de Campo Grande. "
            "PAC (Correios) 5 a 12 dias úteis. SEDEX (Correios) 2 a 5 dias úteis. "
            "Jadlog 3 a 8 dias úteis."
        )},
    ]

    answer = finalize_policy_answer(
        "Para outras cidades, sempre são 2 a 5 dias úteis para o produto 81.",
        evidence,
        "Qual prazo para entrega?",
    )

    assert answer == delivery_time_summary(evidence)
    assert "1 a 3 dias úteis" in answer
    assert "PAC: 5 a 12 dias úteis" in answer
    assert "SEDEX: 2 a 5 dias úteis" in answer
    assert "Jadlog: 3 a 8 dias úteis" in answer
    assert "página 4" in answer and "página 5" in answer
    assert "produto 81" not in answer


def test_sedex_answer_is_limited_to_outside_metropolitan_area():
    evidence = [{"page": 5, "text": (
        "Envios para fora da região metropolitana de Campo Grande. "
        "PAC (Correios) 5 a 12 dias úteis. SEDEX (Correios) 2 a 5 dias úteis. "
        "Jadlog 3 a 8 dias úteis."
    )}]

    answer = shipping_mode_summary(evidence, requested_shipping_modes("quero sedex"))

    assert "SEDEX: 2 a 5 dias úteis" in answer
    assert "fora da região metropolitana" in answer
    assert "página 5, seção 5.2" in answer
    assert "não selecionar o frete" in answer
    assert "PAC:" not in answer
