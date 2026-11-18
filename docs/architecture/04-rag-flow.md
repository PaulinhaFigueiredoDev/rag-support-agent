# Como funciona o assistente da loja

## Objetivo

Mostrar o caminho de uma pergunta até a resposta.

```mermaid
flowchart TB
    classDef person fill:#334155,stroke:#94a3b8,color:#fff
    classDef app fill:#17365d,stroke:#60a5fa,color:#fff
    classDef data fill:#14532d,stroke:#4ade80,color:#fff
    classDef rag fill:#4c1d67,stroke:#d8b4fe,color:#fff
    classDef file fill:#374151,stroke:#cbd5e1,color:#fff

    subgraph CONVERSA["1. Durante o atendimento"]
      direction TB
      P["A pessoa faz uma pergunta"]:::person --> A["Assistente recebe a pergunta<br/>e considera a conversa anterior"]:::app
      A --> D{"O que a pessoa quer saber?"}:::app
      D -->|"Produtos, preços ou pedidos"| T["Consulta as informações da loja"]:::data
      T --> CSV[("Arquivos com dados da loja<br/>CSV")]:::file
      CSV --> R["Assistente prepara a resposta"]:::app
      D -->|"Trocas, devoluções ou entregas"| B["Busca trechos das políticas"]:::rag
      B --> I[("Índice de busca + trechos do manual<br/>FAISS e arquivos JSON")]:::rag
      I --> R
      R --> F["A pessoa recebe a resposta"]:::person
    end

    subgraph PREPARO["2. Antes do atendimento — preparação das políticas"]
      direction LR
      PDF["Manual de políticas<br/>PDF"]:::file --> PRE["Extrai o texto e divide<br/>em pequenos trechos"]:::rag
      PRE --> V["Cria representações do texto<br/>com Ollama"]:::rag
      V --> IDX["Salva o índice de busca<br/>e os trechos"]:::rag
    end
    IDX -.->|"Preparado antes; usado durante a conversa"| I
```

## Entenda em poucas palavras

- **Pergunta:** a pessoa conversa com o assistente pelo terminal.
- **Informações da loja:** quando a pergunta é sobre produtos, preços ou pedidos, o assistente consulta os dados em arquivos CSV.
- **Políticas da loja:** quando a pergunta é sobre trocas, devoluções ou entregas, o assistente pesquisa trechos já preparados do manual.
- **Resposta:** o assistente utiliza as informações encontradas e as regras da aplicação para responder. Se não houver informação suficiente, pode orientar a pessoa a procurar atendimento humano.

## O que acontece antes da conversa?

O manual em PDF é lido **na preparação do sistema**. Seu texto é dividido em trechos e transformado em um índice de busca, com apoio de um modelo de embeddings executado pelo Ollama. **Durante o atendimento, a busca utiliza o índice FAISS e os arquivos com trechos e metadados; ela não abre o PDF novamente.**

> **Simplificação intencional:** o diagrama não detalha cada ferramenta, as chamadas ao modelo, as regras específicas de devolução e frete, nem o limite de quatro chamadas de ferramentas do fluxo principal. Esses detalhes continuam implementados no código.
