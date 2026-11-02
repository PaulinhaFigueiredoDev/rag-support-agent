import json
from pathlib import Path

import faiss
import numpy as np
from ollama import embed

from emporio_assistant.rag.ingestion import extract_pages, make_chunks

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "data" / "documents" / "politicas_da_loja.pdf"
OUTPUT = ROOT / "storage" / "faiss"
MODEL = "embeddinggemma"


def main() -> None:
    chunks = make_chunks(extract_pages(DOCUMENT))
    if not chunks:
        raise ValueError(f"Nenhum texto extraído do PDF: {DOCUMENT}")

    texts = [item["text"] for item in chunks]
    response = embed(model=MODEL, input=texts)
    vectors = np.asarray(response.embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

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
