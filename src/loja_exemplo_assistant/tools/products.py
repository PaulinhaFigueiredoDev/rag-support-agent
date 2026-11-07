from loja_exemplo_assistant.services.data_loader import load_table


def search_products(
    category: str | None = None,
    max_price: float | None = None,
    limit: int = 5,
) -> list[dict]:

    # Valida os parâmetros recebidos
    if not 1 <= limit <= 20:
        raise ValueError("limit deve estar entre 1 e 20")

    if max_price is not None and max_price < 0:
        raise ValueError("max_price deve ser não negativo")

    # Carrega as tabelas da Etapa 2
    products = load_table("products")
    categories = load_table("categories")

    # Relaciona produtos às suas categorias
    table = products.merge(
        categories[["category_id", "name"]],
        on="category_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_category"),
    )

    # Mantém apenas produtos ativos e disponíveis
    table = table.loc[
        (table["status"] == "active") &
        (table["stock_quantity"] > 0)
    ]

    # Filtra pela categoria, quando informada
    if category:
        normalized_category = category.strip().casefold()
        normalized_category = {
            "violão": "violões",
            "violao": "violões",
            "guitarra": "guitarras",
            "baixo": "baixos",
            "ukulele": "ukuleles",
        }.get(normalized_category, normalized_category)
        table = table.loc[
            table["name_category"].str.casefold() == normalized_category
        ]

    # Filtra pelo preço máximo, quando informado
    if max_price is not None:
        table = table.loc[
            table["price_brl"] <= max_price
        ]

    # Ordena pelo preço e limita os resultados
    table = table.sort_values("price_brl").head(limit)

    # Retorna os campos necessários à ferramenta
    return table[
        ["product_id", "name", "price_brl", "stock_quantity"]
    ].to_dict("records")
