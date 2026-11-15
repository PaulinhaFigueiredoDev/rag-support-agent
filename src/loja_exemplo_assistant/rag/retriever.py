import json
import re
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from ollama import Client

embed = Client(timeout=120).embed

ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = ROOT / "storage" / "faiss"
MODEL = "embeddinggemma"
# Calibrado com o manual de demonstração e o índice gerado por embeddinggemma.
MIN_SCORE = 0.40
GENERIC_WORDS = {
    "a", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em", "eu",
    "exemplo", "ha", "loja", "manual", "me", "minha", "no", "na", "nas", "nos",
    "o", "os", "para", "por", "posso", "preciso", "produto", "produtos",
    "qual", "quais", "que", "sao", "se", "sobre", "um", "uma", "voces", "voce",
    "instrumento", "instrumentos", "musica", "musical", "musicais", "politica",
    "politicas", "tem", "ter", "fazem", "oferecem",
}


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        word[:5] for word in re.findall(r"[a-z0-9]+", normalized)
        if word not in GENERIC_WORDS and len(word) >= 3
    }


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
    subject = _terms(question)
    return [
        {"page": chunks[int(i)]["page"], "text": chunks[int(i)]["text"], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i >= 0 and s >= MIN_SCORE and subject & _terms(chunks[int(i)]["text"])
    ]
