# Este script gera o índice FAISS local a partir do manual PDF e dos embeddings do Ollama.

import json
from pathlib import Path

import faiss
import numpy as np
from ollama import embed

from loja_exemplo_assistant.rag.ingestion import extract_pages, make_chunks

ROOT = Path(__file__).resolve().parents[1]
# Define o manual de entrada e o diretório local dos artefatos gerados.
DOCUMENT = ROOT / "data" / "documents" / "politicas_demo.pdf"
OUTPUT = ROOT / "storage" / "faiss"
# Deve ser o mesmo modelo configurado pelo recuperador usado durante as consultas.
MODEL = "embeddinggemma"


# Extrai e divide o manual, cria embeddings e grava índice, trechos e metadados no armazenamento local.
def main() -> None:
    # Extrai texto com número de página e divide o manual em trechos com sobreposição.
    chunks = make_chunks(extract_pages(DOCUMENT))
    if not chunks:
        raise ValueError(f"Nenhum texto extraído do PDF: {DOCUMENT}")

    # Solicita um vetor para cada trecho usando o modelo de embeddings definido acima.
    texts = [item["text"] for item in chunks]
    response = embed(model=MODEL, input=texts)
    vectors = np.asarray(response.embeddings, dtype="float32")
    # Normaliza os vetores para que o produto interno represente similaridade cosseno.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Persiste o índice, os textos paginados e metadados para validar buscas futuras.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT / "index.faiss"))
    (OUTPUT / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {"embedding_model": MODEL, "dimension": vectors.shape[1], "count": len(chunks)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Índice criado: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
