# rag-support-agent

Assistente de atendimento para uma loja de instrumentos musicais, com Ollama local, consultas a dados CSV e busca RAG nas políticas da loja.

## Dados de demonstração

O [manual de políticas](data/documents/politicas_demo.pdf) mantém as regras, seções e páginas do documento usado no desenvolvimento. Nome da loja, razão social, CNPJ, endereço, telefones e e-mail foram substituídos por identificadores fictícios. A loja e os dados de contato do PDF são apenas para demonstração; o CNPJ `00.000.000/0000-00` e o domínio `.invalid` não identificam uma empresa ou canal de atendimento.

O PDF original permanece fora do Git. O índice FAISS também é local e deve ser gerado a partir do manual de demonstração.

## Executar

Requer Python 3.11 ou superior e Ollama em execução. Prepare os modelos e instale o projeto:

```bash
ollama pull qwen2.5:3b
ollama pull embeddinggemma
python -m pip install -e ".[dev]"
cp .env.example .env
```

Gere o índice e inicie a conversa:

```bash
python scripts/build_index.py
python main.py
```

O modelo de chat pode ser ajustado por `OLLAMA_CHAT_MODEL` em `.env`. Para encerrar a conversa, digite `sair`.

## Testes

```bash
python -m pytest -q
```
