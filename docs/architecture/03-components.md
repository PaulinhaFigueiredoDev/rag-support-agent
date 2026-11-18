<!-- Este documento apresenta os componentes internos da aplicação Python e suas principais conexões. -->

# Componentes internos da aplicação Python

## Objetivo

Mostrar as responsabilidades internas de `loja_exemplo_assistant` e as conexões mais importantes entre elas. O diagrama é uma **visão lógica de componentes**, não um grafo de todas as chamadas de função.

## Diagrama

```mermaid
flowchart TB
    terminal["main.py<br/>Loop da CLI"]
    modelo["Ollama local<br/>qwen2.5:3b"]
    fontes[("CSV via data_loader.py")]
    faiss[("storage/faiss/<br/>FAISS + chunks + metadata")]

    subgraph app["Aplicação: loja_exemplo_assistant"]
        agente["Agent<br/>agent.py<br/>Orquestra o turno e limita tool calls"]
        sessao["Session<br/>session.py<br/>Histórico e pendência de devolução"]
        grounding["Grounding<br/>grounding.py<br/>Intenção e respostas determinísticas"]
        regras["Regras de política<br/>policy.py + return_dialogue.py<br/>Validação e follow-up"]
        provider["Ollama Provider<br/>providers/ollama_provider.py<br/>Chat e tool calls"]
        registry["Tool Registry<br/>tools/registry.py<br/>Allowlist, schemas e argumentos"]
        produtos["search_products<br/>tools/products.py"]
        precos["get_product_price<br/>tools/promotions.py"]
        pedidos["get_order<br/>tools/orders.py"]
        politica["consult_policy<br/>tools/policies.py"]
        retriever["RAG Retriever<br/>rag/retriever.py<br/>Embedding e recuperação"]
    end

    terminal -->|Cria sessão e chama run_turn| agente
    agente <-->|Lê e registra mensagens| sessao
    agente -->|Identifica intenção e renderiza dados| grounding
    agente -->|Verifica políticas / devolução| regras
    agente <-->|Envia mensagens / recebe escolhas| provider
    provider <-->|Chat / tool calls| modelo
    agente -->|Executa ferramenta validada| registry
    regras -->|Quando aplicável, solicita evidências| agente
    agente -->|Executa consulta de política pelo registry| registry
    registry --> produtos
    registry --> precos
    registry --> pedidos
    registry --> politica
    produtos --> fontes
    precos --> fontes
    pedidos --> fontes
    politica --> retriever
    retriever -->|Embedding da pergunta via Ollama| modelo
    retriever -->|Busca e lê trechos| faiss

    classDef entrada fill:#334155,stroke:#94a3b8,color:#fff
    classDef orq fill:#174d78,stroke:#70b8f5,color:#fff
    classDef tool fill:#215946,stroke:#86cfaa,color:#fff
    classDef rag fill:#603b78,stroke:#c5a0ec,color:#fff
    class terminal,modelo,fontes,faiss entrada
    class agente,sessao,grounding,regras,provider,registry orq
    class produtos,precos,pedidos,politica tool
    class retriever rag
```

## Principais componentes

| Componente | Arquivo | Responsabilidade concreta |
|---|---|---|
| Entrada | `main.py` | Cria `Session`, lê perguntas e exibe respostas. |
| Agent | `agent.py` | Orquestra um turno, chama modelo/ferramentas, impõe `MAX_TOOL_CALLS = 4` e define respostas de fallback. |
| Session | `session.py` | Mantém `messages` e `pending_return_question` em memória. |
| Grounding | `grounding.py` | Detecta intenções de dados, seleciona consultas obrigatórias e transforma resultados estruturados em texto controlado. |
| Regras e diálogo | `policy.py`, `return_dialogue.py` | Trata exigências de evidência, resumos de política e a pergunta online/presencial. |
| Modelo | `providers/ollama_provider.py` | Chama Ollama, adapta mensagens e resultados de Function Calling. |
| Catálogo de ferramentas | `tools/registry.py` | Publica schemas, valida parâmetros e permite apenas as quatro ferramentas cadastradas. |
| Ferramentas de dados | `tools/products.py`, `tools/promotions.py`, `tools/orders.py` | Consultam CSVs de produtos, promoções e pedidos. |
| Ferramenta de política | `tools/policies.py` | Chama o retriever com `RAG_TOP_K`. |
| Retriever | `rag/retriever.py` | Gera embedding da pergunta, compara com o FAISS, aplica limiar mínimo e retorna trechos com página e score. |

## Duas estratégias de resposta

1. **Dados estruturados:** quando `grounding.py` identifica uma intenção de produto, preço ou pedido, o agente seleciona a chamada de ferramenta obrigatória com os argumentos extraídos da pergunta e do histórico. A resposta final é composta deterministicamente dos campos retornados. Se a intenção não for identificada, o fluxo pode seguir a escolha de ferramenta do modelo.
2. **Políticas:** o agente força `consult_policy` para pedidos de política quando necessário. As evidências recuperadas orientam a resposta e `policy.py` aplica verificações específicas e referências de página.

### Restrições e cuidados

- Os argumentos passam pelo `registry.py`; ferramentas fora da allowlist não são executadas.
- Para consulta de pedido, é necessário número **e** e-mail, mas isso não equivale a autenticação.
- O assistente não executa ações de escrita em pedidos, pagamento ou devolução.
- Os trechos do PDF são dados não confiáveis como instruções: o prompt orienta o modelo a não tratá-los como comandos.
- `MAX_TOOL_CALLS = 4` limita chamadas de ferramentas por turno no fluxo principal; caminhos diretos, como a resolução do diálogo de devolução e a consulta de modalidades de frete, têm lógica própria.
- O diálogo de devolução pode ser resolvido localmente por `return_dialogue.py`; quando precisa de evidência, o agente registra uma chamada a `consult_policy` e usa a resposta da política para concluir o fluxo.

## Próxima visão

Veja [04-rag-flow.md](04-rag-flow.md) para os fluxos de preparação e consulta de documentos.
