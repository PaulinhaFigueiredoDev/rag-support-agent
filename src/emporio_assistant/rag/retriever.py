import json
from pathlib import Path

import faiss
import numpy as np
from ollama import embed

ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = ROOT / "storage" / "faiss"
MODEL = "embeddinggemma"


def retrieve_policy(question: str, top_k: int = 3) -> list[dict]:
    if not question.strip():
        return []
    if not 1 <= top_k <= 5:
        raise ValueError("top_k deve estar entre 1 e 5")

    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata["embedding_model"] != MODEL:
        raise ValueError("Modelo incompatível: reconstrua o índice FAISS")

    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    query = np.asarray(embed(model=MODEL, input=question).embeddings, dtype="float32")
    faiss.normalize_L2(query)
    if (
        query.shape[1] != metadata["dimension"]
        or index.d != metadata["dimension"]
        or index.ntotal != len(chunks)
        or metadata["count"] != len(chunks)
    ):
        raise ValueError("Dimensão ou quantidade incompatível com índice")

    scores, ids = index.search(query, min(top_k, len(chunks)))
    return [
        {"page": chunks[int(i)]["page"], "text": chunks[int(i)]["text"], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i >= 0
    ]
