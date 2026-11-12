from loja_exemplo_assistant.policy import online_return_summary


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
