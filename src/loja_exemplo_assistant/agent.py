"""Orquestra um turno de atendimento com o modelo local."""

from loja_exemplo_assistant.grounding import (
    DATA_TOOLS, data_intent, render_data_answer, required_data_call,
)
from loja_exemplo_assistant.policy import (
    finalize_policy_answer,
    is_policy_request,
    requested_shipping_modes,
    shipping_mode_summary,
)
from loja_exemplo_assistant.prompts import (
    FALLBACK, MODEL_UNAVAILABLE, POLICY_NOT_FOUND, PURCHASE_QUESTION,
)
from loja_exemplo_assistant.providers import ollama_provider
from loja_exemplo_assistant.providers.ollama_provider import ModelUnavailableError, ToolCall
from loja_exemplo_assistant.return_dialogue import resolve_return, session_from_messages
from loja_exemplo_assistant.session import Session
from loja_exemplo_assistant.tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS, execute_tool

# Limita chamadas por turno para evitar ciclos longos ou repetidos de ferramentas.
MAX_TOOL_CALLS = 4


def _required_tool(user_text: str, used_tools: set[str]) -> tuple[str, dict] | None:
    # Garante que perguntas sobre políticas consultem a fonte documental.
    if is_policy_request(user_text) and "consult_policy" not in used_tools:
        return "consult_policy", {"question": user_text}
    return None


def _successful_result(name: str, result: object) -> bool:
    # Uma consulta de política sem trechos também é considerada sem sucesso.
    return not (isinstance(result, dict) and "error" in result) and (name != "consult_policy" or bool(result))


def run_turn(session: Session, user_text: str) -> str:
    """Conduz uma interação e atualiza o histórico da sessão."""
    # Resolve primeiro devoluções que dependem de confirmar o canal de compra.
    # Esse fluxo pode responder ou pedir esclarecimento sem chamar o modelo.
    local = resolve_return(session, user_text)
    if local is not None:
        # Registra pergunta e evidência para manter o histórico coerente com a resposta.
        session.add_user(user_text)
        if local.evidence and local.lookup_question:
            arguments = {"question": local.lookup_question}
            call = ToolCall("consult_policy", arguments)
            session.messages.append(ollama_provider.forced_call_message(call))
            session.messages.append(ollama_provider.tool_result_message(call, local.evidence))
        session.add_assistant(local.answer)
        return local.answer

    # Classifica consultas que precisam usar dados estruturados em vez de confiar no modelo.
    intent = data_intent(session.messages, user_text)
    # Prioriza respostas determinísticas para fretes específicos citados pelo cliente.
    modes = requested_shipping_modes(user_text) if intent != "get_order" else ()
    if modes:
        # Responde sobre modalidades de frete diretamente com evidências da política.
        session.add_user(user_text)
        lookup_question = "Quais são os prazos de entrega de PAC, SEDEX e Jadlog para outras cidades?"
        arguments = {"question": lookup_question}
        evidence = execute_tool("consult_policy", arguments)
        if not isinstance(evidence, list):
            answer = FALLBACK
        elif not evidence:
            answer = POLICY_NOT_FOUND
        else:
            answer = shipping_mode_summary(evidence, modes) or FALLBACK
            call = ToolCall("consult_policy", arguments)
            session.messages.append(ollama_provider.forced_call_message(call))
            session.messages.append(ollama_provider.tool_result_message(call, evidence))
        session.add_assistant(answer)
        return answer

    # Em perguntas de política, limita o contexto ao turno atual para evitar evidência antiga.
    policy_request = is_policy_request(user_text) and intent is None
    # Marca o início do turno para separar contexto novo do histórico anterior.
    turn_start = len(session.messages)
    session.add_user(user_text)
    calls_used = 0
    # Acompanha ferramentas usadas e o estado da recuperação de evidências de política.
    used_tools: set[str] = set()
    policy_evidence: list[dict] = []
    policy_lookup_empty = False
    policy_lookup_failed = False

    while True:
        try:
            model_messages = session.messages
            if policy_request:
                # Preserva a instrução do sistema, mas não mistura conversas anteriores na consulta.
                system = session.messages[:1] if session.messages and session.messages[0]["role"] == "system" else []
                model_messages = system + session.messages[turn_start:]
            response = ollama_provider.complete(model_messages, TOOL_SCHEMAS)
        except ModelUnavailableError:
            # Mantém o histórico consistente mesmo quando o Ollama está indisponível.
            session.add_assistant(MODEL_UNAVAILABLE)
            return MODEL_UNAVAILABLE
        if not response.calls:
            if intent:
                # Para dados de produto e pedido, a aplicação escolhe a ferramenta e os argumentos.
                if calls_used >= MAX_TOOL_CALLS:
                    session.add_assistant(FALLBACK)
                    return FALLBACK
                selected = required_data_call(intent, session.messages, user_text)
                if isinstance(selected, str):
                    session.add_assistant(selected)
                    return selected
                name, arguments = selected
                result = execute_tool(name, arguments)
                if not _successful_result(name, result):
                    # Não substitui uma consulta de dados que falhou por uma afirmação do modelo.
                    session.add_assistant(FALLBACK)
                    return FALLBACK
                call = ToolCall(name, arguments)
                session.messages.append(ollama_provider.forced_call_message(call))
                session.messages.append(ollama_provider.tool_result_message(call, result))
                answer = render_data_answer(name, arguments, result)
                session.add_assistant(answer)
                return answer

            required = _required_tool(user_text, used_tools)
            if required and calls_used < MAX_TOOL_CALLS:
                # Executa a consulta documental obrigatória quando o modelo não a solicitou.
                name, arguments = required
                result = execute_tool(name, arguments)
                if not _successful_result(name, result):
                    answer = POLICY_NOT_FOUND if name == "consult_policy" and result == [] else FALLBACK
                    session.add_assistant(answer)
                    return answer
                call = ToolCall(name, arguments)
                session.messages.append(ollama_provider.forced_call_message(call))
                session.messages.append(ollama_provider.tool_result_message(call, result))
                if name == "consult_policy":
                    policy_evidence = result
                used_tools.add(name)
                calls_used += 1
                continue

            answer = response.content or FALLBACK
            if not policy_evidence and (is_policy_request(user_text) or policy_lookup_empty or policy_lookup_failed):
                # Sem evidência recuperada, não apresenta uma resposta livre sobre políticas.
                answer = POLICY_NOT_FOUND if policy_lookup_empty else FALLBACK
            answer = finalize_policy_answer(answer, policy_evidence, user_text)
            session.add_assistant(answer)
            return answer

        if calls_used + len(response.calls) > MAX_TOOL_CALLS:
            # Recusa o lote inteiro se ele ultrapassar o limite definido para este turno.
            session.add_assistant(FALLBACK)
            return FALLBACK

        selected_calls = []
        # Evita executar mais de uma consulta de dados redundante no mesmo lote.
        data_call_selected = False
        for call in response.calls:
            if call.name in DATA_TOOLS:
                # O modelo pode sugerir a consulta, mas os argumentos vêm da intenção validada.
                if intent is None:
                    session.add_assistant(FALLBACK)
                    return FALLBACK
                if data_call_selected:
                    continue
                selected = required_data_call(intent, session.messages, user_text)
                if isinstance(selected, str):
                    session.add_assistant(selected)
                    return selected
                name, arguments = selected
                selected_calls.append(ToolCall(name, arguments))
                data_call_selected = True
            else:
                selected_calls.append(call)

        data_answers = []
        # Executa as chamadas permitidas e mantém cada resultado no histórico da conversa.
        for call in selected_calls:
            session.messages.append(ollama_provider.forced_call_message(call))
            result = execute_tool(call.name, call.arguments)
            if call.name in DATA_TOOLS:
                if not _successful_result(call.name, result):
                    session.messages.append(ollama_provider.tool_result_message(call, result))
                    session.add_assistant(FALLBACK)
                    return FALLBACK
                data_answers.append(render_data_answer(call.name, call.arguments, result))
            if _successful_result(call.name, result):
                used_tools.add(call.name)
                if call.name == "consult_policy":
                    policy_evidence = result
            elif call.name == "consult_policy":
                # Distingue ausência de trechos de falhas técnicas para escolher a resposta segura.
                policy_lookup_empty = policy_lookup_empty or result == []
                policy_lookup_failed = policy_lookup_failed or result != []
            session.messages.append(ollama_provider.tool_result_message(call, result))
        calls_used += len(response.calls)
        if data_answers:
            answer = "\n\n".join(data_answers)
            session.add_assistant(answer)
            return answer


def run_ollama_turn(messages: list[dict], user_text: str) -> str:
    """Compatibilidade com a API anterior baseada em uma lista de mensagens."""
    return run_turn(session_from_messages(messages), user_text)
