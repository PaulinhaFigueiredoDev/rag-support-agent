"""Seleciona consultas de dados e apresenta apenas valores retornados pelos CSVs."""

import json
import re
from decimal import Decimal

from loja_exemplo_assistant.policy import is_policy_request, normalize_text
from loja_exemplo_assistant.prompts import FALLBACK

# Ferramentas cujo resultado precisa vir dos dados estruturados da loja.
DATA_TOOLS = {"search_products", "get_product_price", "get_order"}
# Perguntas usadas quando faltam informações para fazer uma consulta segura.
ORDER_DETAILS_QUESTION = "Para consultar o pedido, informe o número e o e-mail usado na compra."
PRODUCT_ID_QUESTION = "Para consultar o preço, informe o ID do produto."
STOCK_PRODUCT_QUESTION = "De qual produto você quer consultar o estoque? Informe o ID."
PRICE_LIMIT_QUESTION = "Não entendi o valor máximo. Informe um valor em reais, como R$ 1.000,00."

# Padrões para localizar identificadores mencionados pelo cliente no histórico.
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
ORDER_PATTERN = re.compile(r"\bpedido\s*(?:n[úu]mero\s*)?#?\s*(\d+)\b", re.IGNORECASE)
PRODUCT_PATTERN = re.compile(r"\b(?:produto|id)\s*#?\s*(\d+)\b", re.IGNORECASE)
# Relaciona variações de categoria escritas pelo cliente aos nomes usados nos CSVs.
CATEGORIES = {
    "violao": "Violões", "violoes": "Violões",
    "guitarra": "Guitarras", "guitarras": "Guitarras",
    "baixo": "Baixos", "baixos": "Baixos",
    "ukulele": "Ukuleles", "ukuleles": "Ukuleles",
    "bateria": "Baterias e Percussão", "baterias": "Baterias e Percussão",
    "teclado": "Teclados e Pianos", "teclados": "Teclados e Pianos",
}


# Extrai do histórico apenas as mensagens escritas pelo cliente.
def _user_texts(messages: list[dict]) -> list[str]:
    return [item.get("content", "") for item in messages if item.get("role") == "user"]


def _last_match(pattern: re.Pattern, messages: list[dict]) -> str | None:
    # Procura do fim para o início para priorizar o valor mais recente informado.
    for text in reversed(_user_texts(messages)):
        match = pattern.search(text)
        if match:
            return match.group(1) if pattern.groups else match.group()
    return None


def _last_assistant_text(messages: list[dict]) -> str:
    # A última resposta ajuda a interpretar mensagens curtas, como um ID isolado.
    return next(
        (item.get("content", "") for item in reversed(messages) if item.get("role") == "assistant"),
        "",
    )


def _recent_product_id(messages: list[dict]) -> int | None:
    """Lê o último resultado de ferramenta quando ele identifica um único produto."""
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        return None
    tool_message = messages[-2]
    if tool_message.get("role") != "tool":
        return None
    try:
        result = json.loads(tool_message["content"])
    except (KeyError, TypeError, ValueError):
        return None

    if tool_message.get("tool_name") == "search_products" and isinstance(result, list):
        # Uma busca com vários produtos não permite inferir a qual deles o cliente se refere.
        product_ids = {
            item["product_id"] for item in result
            if isinstance(item, dict) and type(item.get("product_id")) is int
        }
        return next(iter(product_ids)) if len(product_ids) == 1 else None

    # Para preço, recupera o ID dos argumentos da chamada que gerou o resultado.
    if tool_message.get("tool_name") != "get_product_price" or not isinstance(result, dict) or not result.get("found"):
        return None
    call_message = messages[-3]
    if call_message.get("role") != "assistant":
        return None
    try:
        function = call_message["tool_calls"][0]["function"]
        if function["name"] != "get_product_price":
            return None
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        product_id = arguments["product_id"]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return product_id if type(product_id) is int else None


def data_intent(messages: list[dict], user_text: str) -> str | None:
    """Identifica pedidos de dados que não podem ser respondidos sem ferramenta."""
    text = normalize_text(user_text)
    if re.search(r"\b(rastreio|rastreamento|meu pedido|status do pedido)\b", text) or ORDER_PATTERN.search(user_text):
        return "get_order"
    # Perguntas sobre regras da loja seguem o fluxo RAG, não as ferramentas de CSV.
    if is_policy_request(user_text):
        return None
    if re.search(r"\bpagamento\b", text) and not PRODUCT_PATTERN.search(user_text):
        return None
    # Pedidos de sugestões ou listas exigem uma busca de produtos.
    if re.search(r"\b(mostre|liste|sugira|indique|recomende|procuro|busco|quais|opcoes)\b", text) and re.search(
        r"\b(produto|produtos|instrumento|instrumentos|violao|violoes|guitarra|guitarras|baixo|baixos|ukulele|ukuleles|bateria|baterias|teclado|teclados)\b",
        text,
    ):
        return "search_products"
    if re.search(r"\b(precos?|valores?|custas?|custam|descontos?|promocao|promocoes|pix)\b", text):
        return "get_product_price"
    # Perguntas sobre disponibilidade consultam a quantidade de estoque nos CSVs.
    if re.search(r"\b(estoque|disponivel|disponiveis)\b", text):
        return "search_products"
    if re.search(r"\b(tem|vende|vendem)\b", text) and re.search(
        r"\b(produto|produtos|instrumento|instrumentos|violao|violoes|guitarra|guitarras|baixo|baixos|ukulele|ukuleles|bateria|baterias|teclado|teclados)\b",
        text,
    ):
        return "search_products"
    # Uma resposta curta pode continuar uma solicitação de dados feita no turno anterior.
    previous = _last_assistant_text(messages)
    if previous == ORDER_DETAILS_QUESTION:
        return "get_order"
    if previous == PRODUCT_ID_QUESTION:
        return "get_product_price"
    if previous == STOCK_PRODUCT_QUESTION:
        return "search_products"
    return None


def _search_arguments(user_text: str, messages: list[dict]) -> dict:
    # Usa cinco resultados por padrão e reduz para um quando o cliente pede um único item.
    text = normalize_text(user_text)
    arguments: dict = {"limit": 5}
    if re.search(
        r"\b(?:um|uma)\s+(?:produto|instrumento|violao|guitarra|baixo|ukulele|bateria|teclado)\b",
        text,
    ):
        arguments["limit"] = 1
    for word, category in CATEGORIES.items():
        if re.search(rf"\b{word}\b", text):
            arguments["category"] = category
            break
    # Converte limites monetários brasileiros, como R$ 1.000,00, para número decimal.
    price_match = re.search(r"\b(?:ate|abaixo de|menos de)\s*(?:r\$\s*)?(\d[\d.,]*)", text)
    if price_match:
        number = price_match.group(1).rstrip(".,")
        if "," in number:
            number = number.replace(".", "").replace(",", ".")
        # Um ponto seguido por três dígitos é interpretado como separador de milhar.
        elif number.count(".") > 1 or ("." in number and len(number.rsplit(".", 1)[-1]) == 3):
            number = number.replace(".", "")
        try:
            arguments["max_price"] = float(number)
        except ValueError:
            raise ValueError(PRICE_LIMIT_QUESTION) from None
    product_id = _last_match(PRODUCT_PATTERN, messages)
    if product_id and PRODUCT_PATTERN.search(user_text):
        arguments["product_id"] = int(product_id)
    return arguments


def required_data_call(intent: str, messages: list[dict], user_text: str) -> tuple[str, dict] | str:
    # Monta a chamada necessária e pede ao cliente os dados obrigatórios ausentes.
    if intent == "get_order":
        order_id = _last_match(ORDER_PATTERN, messages)
        email = _last_match(EMAIL_PATTERN, messages)
        if not order_id or not email:
            return ORDER_DETAILS_QUESTION
        return "get_order", {"order_id": int(order_id), "email": email}
    if intent == "get_product_price":
        previous = _last_assistant_text(messages[:-1])
        product_id = _last_match(PRODUCT_PATTERN, messages)
        if not product_id:
            # Aceita um ID isolado quando o agente acabou de solicitá-lo.
            if previous == PRODUCT_ID_QUESTION and re.fullmatch(r"\s*\d+\s*", user_text):
                product_id = user_text.strip()
        if not product_id:
            return PRODUCT_ID_QUESTION
        payment = None
        if "pix" in normalize_text(user_text):
            payment = "pix"
        elif previous == PRODUCT_ID_QUESTION and len(_user_texts(messages)) > 1:
            # Preserva o Pix informado antes da pergunta de esclarecimento do ID.
            prior_question = _user_texts(messages)[-2]
            if "pix" in normalize_text(prior_question):
                payment = "pix"
        return "get_product_price", {"product_id": int(product_id), "payment": payment}
    try:
        arguments = _search_arguments(user_text, messages)
    except ValueError:
        return PRICE_LIMIT_QUESTION
    previous = _last_assistant_text(messages[:-1])
    # Usa como ID a resposta numérica dada após perguntar qual produto consultar.
    if previous == STOCK_PRODUCT_QUESTION and re.fullmatch(r"\s*\d+\s*", user_text):
        arguments["product_id"] = int(user_text.strip())
    text = normalize_text(user_text)
    asks_stock = re.search(r"\b(estoque|disponivel|disponiveis)\b", text) or previous == STOCK_PRODUCT_QUESTION
    is_general_list = re.search(
        r"\b(produtos|instrumentos|mostre|liste|sugira|indique|recomende|quais|opcoes)\b",
        text,
    )
    # Não adivinha o produto quando a pergunta de estoque não identifica um item único.
    if asks_stock and not is_general_list and "product_id" not in arguments and "category" not in arguments:
        product_id = _recent_product_id(messages[:-1])
        if product_id is None:
            return STOCK_PRODUCT_QUESTION
        arguments["product_id"] = product_id
    return "search_products", arguments


def _money(value: object) -> str:
    # Formata valores numéricos no padrão monetário brasileiro.
    number = Decimal(str(value))
    return "R$ " + f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_data_answer(name: str, arguments: dict, result: object) -> str:
    """Ignora a redação do modelo e usa somente os campos retornados pela ferramenta."""
    try:
        if name == "search_products":
            if not isinstance(result, list):
                return FALLBACK
            if not result:
                return "Não encontrei produtos ativos em estoque para essa busca."
            # Monta cada linha com os campos de produto retornados pela ferramenta.
            lines = [
                f"- {item['name']} (ID {item['product_id']}): {_money(item['price_brl'])}; "
                f"{item['stock_quantity']} em estoque."
                for item in result
            ]
            return "Produtos disponíveis:\n" + "\n".join(lines)
        if name == "get_product_price":
            # Só informa valores quando a ferramenta confirmou que encontrou o produto.
            if not isinstance(result, dict) or not result.get("found"):
                return "Não encontrei esse produto na base de preços."
            return (
                f"Produto {arguments['product_id']}: preço original {_money(result['original_price'])}; "
                f"desconto de {result['discount_percent']}%; preço final {_money(result['final_price'])}."
            )
        if name == "get_order":
            # Traduz status internos e inclui rastreio ou previsão apenas quando disponíveis.
            if not isinstance(result, dict) or not result.get("found"):
                return "Não encontrei pedido correspondente a esse número e e-mail."
            status = {
                "pending": "pendente", "processing": "em processamento", "shipped": "enviado",
                "delivered": "entregue", "cancelled": "cancelado",
            }.get(result["status"], result["status"])
            answer = f"Pedido {result['order_id']}: status {status}."
            if result.get("tracking_code"):
                answer += f" Código de rastreio: {result['tracking_code']}."
            if result.get("estimated_delivery"):
                answer += f" Previsão cadastrada: {result['estimated_delivery']}."
            return answer
    # Dados incompletos ou incompatíveis geram uma resposta segura ao cliente.
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return FALLBACK
    return FALLBACK
