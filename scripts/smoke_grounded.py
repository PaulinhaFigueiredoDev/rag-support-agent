"""Teste manual de ponta a ponta das respostas baseadas nos CSVs.

Execute na raiz do projeto com Ollama, modelos de chat e embedding e índice FAISS prontos:
    python scripts/smoke_grounded.py
"""

from loja_exemplo_assistant.agent import run_turn
from loja_exemplo_assistant.prompts import MODEL_UNAVAILABLE
from loja_exemplo_assistant.session import Session


def check(session: Session, question: str, expected: str | tuple[str, ...], forbidden: str | None = None) -> None:
    answer = run_turn(session, question)
    print(f"Cliente: {question}\nAgente: {answer}\n")
    if answer == MODEL_UNAVAILABLE:
        raise SystemExit("O Ollama não respondeu. Verifique se ele está ativo e se o modelo foi instalado.")
    expected_items = (expected,) if isinstance(expected, str) else expected
    for item in expected_items:
        if item not in answer:
            raise SystemExit(f"Falhou: esperava encontrar {item!r} na resposta.")
    if forbidden and forbidden in answer:
        raise SystemExit(f"Falhou: a resposta contém {forbidden!r}.")


def main() -> None:
    product_session = Session()
    check(product_session, "Qual o preço do produto 81 no Pix?", "R$ 569,91")
    check(product_session, "esta disponivel em estoque?", "12 em estoque")
    check(Session(), "O produto 81 está disponível em estoque?", "12 em estoque")

    order_session = Session()
    check(order_session, "Qual o status do pedido 1?", "informe o número e o e-mail")
    check(order_session, "pedro.oliveira@jmail.com", "Pedido 1: status entregue")
    check(Session(), "Status do pedido 1, e-mail errado@example.invalid", "Não encontrei pedido")

    price_session = Session()
    check(price_session, "Quanto custa no Pix?", "informe o ID do produto")
    check(price_session, "81", "R$ 569,91")
    check(
        price_session,
        "qual prazo para entrega?",
        ("página 4", "página 5", "PAC: 5 a 12 dias úteis", "SEDEX: 2 a 5 dias úteis", "Jadlog: 3 a 8 dias úteis"),
        forbidden="produto 81",
    )
    check(price_session, "quero sedex", ("SEDEX: 2 a 5 dias úteis", "fora da região metropolitana", "página 5"))

    print("Teste concluído: respostas de preço, estoque, pedido e entrega conferidas.")


if __name__ == "__main__":
    main()
