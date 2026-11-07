from decimal import Decimal
from loja_exemplo_assistant.services.data_loader import load_table
from loja_exemplo_assistant.tools.promotions import get_product_price


def test_unknown_product_is_not_found():
    assert get_product_price(-1) == {"found": False}


def test_inactive_promotion_not_applied():
    promotions = load_table("promotions")
    products = load_table("products")
    inactive = promotions.loc[promotions["is_active"] == 0]
    candidates = inactive.loc[~inactive["product_id"].isin(promotions.loc[promotions["is_active"] == 1, "product_id"])]
    candidates = candidates.loc[candidates["product_id"].isin(products["product_id"])]
    assert not candidates.empty, "A base de exemplo precisa de um produto com promoção somente inativa"
    result = get_product_price(int(candidates.iloc[0]["product_id"]))
    assert Decimal(result["discount_percent"]) == Decimal("0")


def test_pix_discount_without_active_promotion():
    promotions = load_table("promotions")
    products = load_table("products")
    active_ids = set(promotions.loc[promotions["is_active"] == 1, "product_id"])
    candidates = products.loc[~products["product_id"].isin(active_ids)]
    assert not candidates.empty
    result = get_product_price(int(candidates.iloc[0]["product_id"]), payment="pix")
    assert Decimal(result["discount_percent"]) == Decimal("5")


def test_pix_does_not_stack_with_active_promotion():
    promotions = load_table("promotions")
    products = load_table("products")
    active = promotions.loc[(promotions["is_active"] == 1) & (promotions["product_id"].isin(products["product_id"]))]
    assert not active.empty
    first = active.iloc[0]
    result = get_product_price(int(first["product_id"]), payment="pix")
    assert Decimal(result["discount_percent"]) == Decimal(str(first["discount_percent"]))