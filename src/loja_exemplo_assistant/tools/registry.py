"""Catálogo permitido, schemas e execução segura das ferramentas."""

import json
import math

from httpx import RequestError as HttpRequestError
from ollama import RequestError, ResponseError

from loja_exemplo_assistant.tools.orders import get_order
from loja_exemplo_assistant.tools.products import search_products
from loja_exemplo_assistant.tools.promotions import get_product_price
from loja_exemplo_assistant.tools.policies import consult_policy

# Lista explícita: o modelo só pode solicitar estas ferramentas registradas.
TOOLS = [search_products, get_order, get_product_price, consult_policy]
TOOL_REGISTRY = {function.__name__: function for function in TOOLS}


# Monta a definição de uma ferramenta no formato esperado pelo provedor do modelo.
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


# O modelo recebe apenas os nomes e formatos declarados nesta lista.
TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "Lista produtos ativos em estoque. Todos os filtros são opcionais; use sem argumentos para sugestões gerais.",
        {
            "category": {"type": ["string", "null"]},
            "max_price": {"type": ["number", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "product_id": {"type": ["integer", "null"]},
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
    # bool é rejeitado explicitamente, apesar de ser uma subclasse de int em Python.
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(name)
    return value


def _text(value: object, name: str) -> str:
    # Exige texto não vazio e limitado a 500 caracteres para entradas de ferramenta.
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError(name)
    return value.strip()


# Rejeita NaN e infinitos, que não fazem parte do padrão JSON.
def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)

# Converte os argumentos do modelo para estruturas e valores conhecidos.
def _validate_arguments(name: str, arguments: object) -> dict:
    # O provedor pode entregar argumentos como objeto Python ou como texto JSON.
    if isinstance(arguments, str):
        arguments = json.loads(arguments, parse_constant=_reject_nonfinite)
    if not isinstance(arguments, dict):
        raise ValueError("arguments")

    # Impede campos extras que não fazem parte dos argumentos aceitos pela ferramenta.
    allowed = {
        "search_products": {"category", "max_price", "limit", "product_id"},
        "get_order": {"order_id", "email"},
        "get_product_price": {"product_id", "payment"},
        "consult_policy": {"question"},
    }[name]
    if arguments.keys() - allowed:
        raise ValueError("arguments")

    if name == "search_products":
        # Valida cada filtro opcional e não inclui filtros que não foram informados.
        result = {}
        if arguments.get("category") is not None:
            result["category"] = _text(arguments["category"], "category")
        if arguments.get("max_price") is not None:
            price = arguments["max_price"]
            # Preço deve ser finito e não negativo; valores booleanos não são aceitos.
            if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
                raise ValueError("max_price")
            result["max_price"] = price
        if "limit" in arguments:
            result["limit"] = _positive_int(arguments["limit"], "limit", 20)
        if arguments.get("product_id") is not None:
            result["product_id"] = _positive_int(arguments["product_id"], "product_id")
        return result
    if name == "get_order":
        # Exige os dois dados usados para localizar o pedido do cliente.
        return {
            "order_id": _positive_int(arguments.get("order_id"), "order_id"),
            "email": _text(arguments.get("email"), "email"),
        }
    if name == "get_product_price":
        # O ID é obrigatório; pagamento é opcional e só entra quando informado.
        result = {"product_id": _positive_int(arguments.get("product_id"), "product_id")}
        if arguments.get("payment") is not None:
            result["payment"] = _text(arguments["payment"], "payment")
        return result
    return {"question": _text(arguments.get("question"), "question")}


def execute_tool(name: str, arguments: object) -> dict | list[dict]:
    # Busca apenas na lista permitida e recusa nomes de ferramenta desconhecidos.
    function = TOOL_REGISTRY.get(name)
    if function is None:
        return {"error": "Ferramenta não permitida"}
    try:
        validated = _validate_arguments(name, arguments)
    except (TypeError, ValueError):
        return {"error": "Argumentos inválidos"}
    try:
        return function(**validated)
    except (TypeError, ValueError, FileNotFoundError, ConnectionError,
            HttpRequestError, RequestError, ResponseError):
        # Evita expor detalhes internos de infraestrutura ao agente ou ao cliente.
        return {"error": "Consulta indisponível"}
