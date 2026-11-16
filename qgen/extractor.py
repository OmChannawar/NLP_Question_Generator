"""
extractor.py
------------
Document ingestion layer: reads PDF / DOCX textbooks, cleans the raw text and
splits it into passages ("chunks") that fit inside the Transformer's input window.
"""
import io
import re
from typing import List

import numpy as np
from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return the concatenated text of every page in a PDF (pages that fail are skipped)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # corrupt / image-only page
            continue
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Return the text of all non-empty paragraphs of a Word (.docx) document."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right extractor based on the file extension (.pdf or .docx)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == "docx":
        return extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Please upload a .pdf or .docx file.")


def clean_text(text: str) -> str:
    """
    Normalise raw textbook text:
      * re-join words hyphenated across line breaks,
      * drop lines that are only a page number,
      * collapse all whitespace into single spaces.
    """
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Light-weight rule-based sentence splitter (no model download needed)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _is_prose(chunk: str) -> bool:
    """Heuristic filter: keep passages that look like sentences, not tables/indexes/references."""
    letters = sum(c.isalpha() or c.isspace() for c in chunk)
    return letters / max(len(chunk), 1) > 0.85 and chunk.count(".") >= 2


def build_chunks(text: str, max_words: int = 170, min_words: int = 50) -> List[str]:
    """
    Group consecutive sentences into passages of roughly `max_words` words.
    Keeping passages short guarantees they fit the 512-token input limit of T5-family models.
    """
    chunks, current, count = [], [], 0
    for sent in split_sentences(text):
        n_words = len(sent.split())
        if current and count + n_words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n_words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= min_words and _is_prose(c)]


def select_chunks(chunks: List[str], n: int) -> List[str]:
    """
    Pick `n` passages spread evenly across the whole book so questions cover all topics.
    If the book has fewer passages than `n`, passages are reused (a different question is
    sampled each time by the generator).
    """
    if len(chunks) >= n:
        idx = np.linspace(0, len(chunks) - 1, n).round().astype(int)
        return [chunks[i] for i in idx]
    return [chunks[i % len(chunks)] for i in range(n)]
