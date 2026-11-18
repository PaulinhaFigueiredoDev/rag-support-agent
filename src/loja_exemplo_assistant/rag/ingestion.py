# Este módulo extrai páginas do PDF e divide o conteúdo em trechos para indexação.

from pathlib import Path

from pypdf import PdfReader


# Extrai texto não vazio do PDF e preserva o número original de cada página.
def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def make_chunks(pages: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
# Divide o texto por página em trechos com sobreposição para preservar contexto entre blocos.
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Tamanho e sobreposição inválidos")

    chunks = []
    step = size - overlap
    for entry in pages:
        text = entry["text"]
        for start in range(0, len(text), step):
            part = text[start:start + size].strip()
            if part:
                chunks.append({"page": entry["page"], "text": part})
            if start + size >= len(text):
                break
    return chunks
