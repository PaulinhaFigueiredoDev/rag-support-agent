import pytest
from emporio_assistant.tools.products import search_products


def test_search_products_applies_filters():
    items = search_products(category="Violões", max_price=1000, limit=20)
    assert all(item["price_brl"] <= 1000 for item in items)
    assert all(item["stock_quantity"] > 0 for item in items)
    assert len(items) <= 20


def test_singular_violao_category_matches_csv_category():
    assert search_products(category="Violão", max_price=1000) == search_products(
        category="Violões", max_price=1000
    )


def test_invalid_limit_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        search_products(limit=0)


def test_negative_price_is_rejected():
    with pytest.raises(ValueError, match="max_price"):
        search_products(max_price=-1)
