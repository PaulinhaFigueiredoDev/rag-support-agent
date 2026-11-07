"""Catálogo permitido, schemas e execução segura das ferramentas."""

import json
import math

from loja_exemplo_assistant.tools.orders import get_order
from loja_exemplo_assistant.tools.products import search_products
from loja_exemplo_assistant.tools.promotions import get_product_price
from loja_exemplo_assistant.tools.policies import consult_policy

TOOLS = [search_products, get_order, get_product_price, consult_policy]
TOOL_REGISTRY = {function.__name__: function for function in TOOLS}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "Lista produtos ativos em estoque. Todos os filtros são opcionais; use sem argumentos para sugestões gerais.",
        {
            "category": {"type": ["string", "null"]},
            "max_price": {"type": ["number", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        [],
    ),
    _schema(
        "get_order",
        "Consulta status, rastreio e previsão de um pedido após receber número e e-mail do cliente.",
        {"order_id": {"type": "integer"}, "email": {"type": "string"}},
        ["order_id", "email"],
    ),
    _schema(
        "get_product_price",
        "Consulta o preço e o desconto de um produto, incluindo pagamento por Pix quando informado.",
        {"product_id": {"type": "integer"}, "payment": {"type": ["string", "null"]}},
        ["product_id"],
    ),
    _schema(
        "consult_policy",
        "Recupera trechos das políticas da loja com número da página.",
        {"question": {"type": "string"}},
        ["question"],
    ),
]


def _positive_int(value: object, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(name)
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError(name)
    return value.strip()


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


def _validate_arguments(name: str, arguments: object) -> dict:
    if isinstance(arguments, str):
        arguments = json.loads(arguments, parse_constant=_reject_nonfinite)
    if not isinstance(arguments, dict):
        raise ValueError("arguments")

    allowed = {
        "search_products": {"category", "max_price", "limit"},
        "get_order": {"order_id", "email"},
        "get_product_price": {"product_id", "payment"},
        "consult_policy": {"question"},
    }[name]
    if arguments.keys() - allowed:
        raise ValueError("arguments")

    if name == "search_products":
        result = {}
        if arguments.get("category") is not None:
            result["category"] = _text(arguments["category"], "category")
        if arguments.get("max_price") is not None:
            price = arguments["max_price"]
            if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
                raise ValueError("max_price")
            result["max_price"] = price
        if "limit" in arguments:
            result["limit"] = _positive_int(arguments["limit"], "limit", 20)
        return result
    if name == "get_order":
        return {
            "order_id": _positive_int(arguments.get("order_id"), "order_id"),
            "email": _text(arguments.get("email"), "email"),
        }
    if name == "get_product_price":
        result = {"product_id": _positive_int(arguments.get("product_id"), "product_id")}
        if arguments.get("payment") is not None:
            result["payment"] = _text(arguments["payment"], "payment")
        return result
    return {"question": _text(arguments.get("question"), "question")}


def execute_tool(name: str, arguments: object) -> dict | list[dict]:
    function = TOOL_REGISTRY.get(name)
    if function is None:
        return {"error": "Ferramenta não permitida"}
    try:
        validated = _validate_arguments(name, arguments)
    except (TypeError, ValueError):
        return {"error": "Argumentos inválidos"}
    try:
        return function(**validated)
    except (TypeError, ValueError, FileNotFoundError, ConnectionError):
        return {"error": "Consulta indisponível"}
