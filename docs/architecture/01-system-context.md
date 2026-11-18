<!-- Este documento descreve o contexto do sistema, seus usuários e as dependências externas à aplicação. -->

# Contexto do sistema

## Objetivo

Mostrar qual é a responsabilidade do sistema e quais dependências estão fora do limite da aplicação.

## Diagrama

```mermaid
flowchart LR
    cliente["Cliente / operador<br/>Pessoa que faz perguntas no terminal"]
    subgraph ambiente["Ambiente local"]
        sistema["RAG Support Agent<br/>Assistente de atendimento em Python"]
        ollama["Ollama<br/>Runtime local de modelos"]
        arquivos[("Arquivos locais<br/>CSV, PDF e índice FAISS")]
    end

    cliente -->|Faz perguntas e lê respostas pela CLI| sistema
    sistema -->|Solicita geração de respostas e embeddings| ollama
    ollama -->|Retorna texto, tool calls e vetores| sistema
    sistema -->|Consulta CSVs e material documental indexado| arquivos

    classDef pessoa fill:#27364a,stroke:#78a9d8,color:#fff
    classDef sistema fill:#174d78,stroke:#70b8f5,color:#fff
    classDef externo fill:#4a3b62,stroke:#be9ce8,color:#fff
    classDef dados fill:#264d43,stroke:#81c9aa,color:#fff
    class cliente pessoa
    class sistema sistema
    class ollama externo
    class arquivos dados
```

## Responsabilidades e fronteiras

| Elemento | Responsabilidade | Limite |
|---|---|---|
| Cliente / operador | Digita perguntas e recebe respostas no terminal. | Fora da aplicação. |
| RAG Support Agent | Coordena o atendimento, consulta ferramentas e responde com base nos dados disponíveis. | Sistema em estudo. |
| Ollama | Executa localmente o modelo de chat (`qwen2.5:3b`) e o modelo de embeddings (`embeddinggemma`). | Runtime separado, embora na mesma máquina. |
| Arquivos locais | Fornecem produtos, pedidos e promoções por CSV; o manual PDF é a fonte para gerar o índice FAISS consultado durante o atendimento. | Persistência em disco, não um serviço externo. |

## O que o sistema faz

- Consulta produtos disponíveis, preços, descontos e status de pedidos.
- Recupera trechos do manual de políticas com indicação de página por meio de RAG.
- Mantém o contexto de conversa em memória apenas durante a sessão em execução.
- Restringe respostas factuais a consultas e evidências recuperadas.

## O que o sistema não faz

Não altera pedidos, processa pagamentos, inicia devoluções, autentica clientes nem utiliza uma API de nuvem para os modelos. A conferência de número de pedido e e-mail serve apenas como filtro da consulta, **não como autenticação**.

## Próxima visão

Veja [02-containers.md](02-containers.md) para entender os processos e armazenamentos locais.
