from emporio_assistant.services.data_loader import load_table


def get_order(order_id: int, email: str) -> dict:
    if order_id < 1 or not email.strip():
        return {"found": False}

    orders = load_table("orders")
    customers = load_table("customers")
    joined = orders.merge(
        customers[["customer_id", "email"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    matched = joined.loc[
        (joined["order_id"] == order_id) &
        (joined["email"].str.casefold() == email.strip().casefold())
    ]
    if matched.empty:
        return {"found": False}

    row = matched.iloc[0]
    return {
        "found": True,
        "order_id": int(row["order_id"]),
        "status": str(row["status"]),
        "tracking_code": None if row["tracking_code"] != row["tracking_code"] else str(row["tracking_code"]),
        "estimated_delivery": None if row["estimated_delivery"] != row["estimated_delivery"] else str(row["estimated_delivery"]),
    }