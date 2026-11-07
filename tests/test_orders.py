from loja_exemplo_assistant.tools.orders import get_order
from loja_exemplo_assistant.services.data_loader import load_table


def test_wrong_email_does_not_disclose_order():
    assert get_order(1, "nao-existe@example.invalid") == {"found": False}


def test_invalid_order_is_not_found():
    assert get_order(-1, "teste@example.invalid") == {"found": False}


def test_matching_order_returns_only_allowed_fields():
    orders = load_table("orders")
    customers = load_table("customers")
    first = orders.merge(customers[["customer_id", "email"]], on="customer_id").iloc[0]
    result = get_order(int(first["order_id"]), str(first["email"]))
    assert result["found"] is True
    assert set(result) == {"found", "order_id", "status", "tracking_code", "estimated_delivery"}