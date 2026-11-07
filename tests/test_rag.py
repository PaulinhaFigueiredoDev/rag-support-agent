import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from loja_exemplo_assistant.rag import ingestion, retriever
from loja_exemplo_assistant.policy import _return_policy_summary


def test_demo_pdf_supports_return_policy():
    document = Path(__file__).resolve().parents[1] / "data" / "documents" / "politicas_demo.pdf"
    pages = ingestion.extract_pages(document)

    assert len(pages) == 8
    page_four = next(item for item in pages if item["page"] == 4)
    assert _return_policy_summary([page_four], confirmed_online=True) is not None


def test_extract_pages_skips_empty_pages(monkeypatch, tmp_path):
    pages = [
        SimpleNamespace(extract_text=lambda: "  Política de devolução  "),
        SimpleNamespace(extract_text=lambda: ""),
    ]
    monkeypatch.setattr(ingestion, "PdfReader", lambda path: SimpleNamespace(pages=pages))

    assert ingestion.extract_pages(tmp_path / "manual.pdf") == [
        {"page": 1, "text": "Política de devolução"}
    ]


def test_chunks_preserve_page_and_overlap():
    pages = [{"page": 4, "text": "abcdefgh"}]

    assert ingestion.make_chunks(pages, size=6, overlap=2) == [
        {"page": 4, "text": "abcdef"},
        {"page": 4, "text": "efgh"},
    ]


@pytest.mark.parametrize("size, overlap", [(0, 0), (6, -1), (6, 6)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError, match="Tamanho e sobreposição"):
        ingestion.make_chunks([], size=size, overlap=overlap)


def test_retriever_checks_arguments_before_loading_index():
    assert retriever.retrieve_policy("   ") == []
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_policy("devolução", top_k=0)


def test_retriever_returns_most_similar_chunk_with_page(tmp_path, monkeypatch):
    chunks = [
        {"page": 2, "text": "Entrega"},
        {"page": 4, "text": "Devolução"},
    ]
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0, 1], [1, 0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    (tmp_path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"embedding_model": "embeddinggemma", "dimension": 2, "count": 2}),
        encoding="utf-8",
    )
    monkeypatch.setattr(retriever, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(
        retriever, "embed", lambda **kwargs: SimpleNamespace(embeddings=[[1, 0]])
    )

    results = retriever.retrieve_policy("Posso devolver?", top_k=1)

    assert results == [{"page": 4, "text": "Devolução", "score": 1.0}]


def test_retriever_rejects_stale_metadata(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"embedding_model": "outro-modelo", "dimension": 2, "count": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(retriever, "INDEX_DIR", tmp_path)

    with pytest.raises(ValueError, match="Modelo incompatível"):
        retriever.retrieve_policy("devolução")
