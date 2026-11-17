# rag-support-agent

Assistente de atendimento para uma loja de instrumentos musicais, com Ollama local, consultas a dados CSV e busca RAG nas políticas da loja.

## Dados de demonstração

O [manual de políticas](data/documents/politicas_demo.pdf) mantém as regras, seções e páginas do documento usado no desenvolvimento. Nome da loja, razão social, CNPJ, endereço, telefones e e-mail foram substituídos por identificadores fictícios. A loja e os dados de contato do PDF são apenas para demonstração; o CNPJ `00.000.000/0000-00` e o domínio `.invalid` não identificam uma empresa ou canal de atendimento.

O PDF original permanece fora do Git. O índice FAISS também é local e deve ser gerado a partir do manual de demonstração.

## Preparar o ambiente

Execute os comandos na raiz do repositório. São necessários Python 3.11 ou superior e [Ollama](https://ollama.com/) em execução. No macOS e Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
ollama pull qwen2.5:3b
ollama pull embeddinggemma
```

No Windows, ative o ambiente com `.venv\Scripts\Activate.ps1` antes da instalação. Se `ollama list` não conseguir conectar, inicie o aplicativo Ollama ou execute `ollama serve` em outro terminal.

Gere o índice local a partir do PDF de demonstração:

```bash
python scripts/build_index.py
```

Os arquivos são criados em `storage/faiss/` e não entram no Git. Refaça esse comando se o PDF ou o modelo de embeddings mudar.

## Conversar com o agente

```bash
python main.py
```

Exemplos de perguntas para os dados de demonstração:

```text
Cliente: Qual o preço do produto 81 no Pix?
Cliente: O produto 81 está disponível em estoque?
Cliente: Qual o status do pedido 1?
Cliente: pedro.oliveira@jmail.com
Cliente: Qual o prazo de entrega?
Cliente: Quero SEDEX
```

Digite `sair` para encerrar. O histórico existe apenas durante a conversa aberta; uma nova execução começa outra sessão.

## Configuração

O arquivo `.env` permite ajustar:

| Variável | Padrão | Uso |
| --- | --- | --- |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Modelo local usado para conversar e escolher ferramentas. Instale o modelo escolhido com `ollama pull`. |
| `RAG_TOP_K` | `3` | Quantidade de trechos candidatos de política por pergunta. Aceita inteiros de `1` a `5`. |

O modelo de embeddings é `embeddinggemma`; o índice FAISS registra esse nome e precisa ser reconstruído se ele for alterado no código. O valor `3` para `RAG_TOP_K` foi usado nos testes de conversa. Valores menores podem deixar de recuperar páginas necessárias para uma resposta completa. Reinicie `main.py` após mudar `.env`.

## Testar

Os testes automatizados usam respostas simuladas do Ollama:

```bash
python -m pytest -q
```

Para conferir uma conversa real com Ollama e o índice FAISS, execute:

```bash
python scripts/smoke_grounded.py
```

O roteiro verifica preço, estoque, pedido com e sem e-mail correto, prazo de entrega e SEDEX. Ele termina com `Teste concluído` quando todas as verificações passam.

## Limites do atendimento

Produtos, preços e pedidos vêm dos CSVs de demonstração; eles não refletem uma loja em operação. O agente consulta o status do pedido após receber número e e-mail, mas essa conferência não autentica a identidade do cliente. Ele não altera pedidos, processa pagamentos, inicia devoluções nem seleciona frete. Para essas ações, oriente o cliente a procurar atendimento humano.
