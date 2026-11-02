from decimal import Decimal, ROUND_HALF_UP
from emporio_assistant.services.data_loader import load_table

def get_product_price(product_id: int, payment: str | None = None) -> dict:
    products = load_table("products")
    promotions = load_table("promotions")
    matches = products.loc[products["product_id"] == product_id]
    if matches.empty:
        return {"found": False}
    product = matches.iloc[0]
    base = Decimal(str(product["price_brl"]))

    active = promotions.loc[
        (promotions["product_id"] == product_id) &
        (promotions["is_active"] == 1)
    ]
    if len(active) > 1:
        raise ValueError("Mais de uma promoção ativa; exige regra de escolha")

    discount = (
        Decimal(str(active.iloc[0]["discount_percent"]))
        if len(active)
        else Decimal("0")
    )
    if discount < 0 or discount > 100:
        raise ValueError("Desconto inválido")
    if discount == 0 and payment and payment.casefold() == "pix":
        discount = Decimal("5")

    final = (base * (Decimal("1") - discount / Decimal("100"))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return {
        "found": True,
        "original_price": str(base),
        "discount_percent": str(discount),
        "final_price": str(final),
    }