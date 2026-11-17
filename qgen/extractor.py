"""
extractor.py
------------
Document ingestion layer: reads PDF / DOCX textbooks and turns them into *section-aware* passages.

Why section-aware?  If a passage straddles two sections, the language model sees unrelated facts side by
side and invents questions that mix them.  Here every passage stays inside one section and carries the
section title, which also helps the model resolve words like "it" or "this".
"""
import io
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from docx import Document
from pypdf import PdfReader


# --------------------------------------------------------------------------- file reading
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return the text of every page of a PDF, one line per text line (pages that fail are skipped)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # corrupt / image-only page
            continue
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Return the text of a Word document, one paragraph per line.
    Paragraphs styled as Heading/Title get a '# ' prefix; other paragraphs get a bullet marker so that
    each paragraph stays a separate unit later on.
    """
    doc = Document(io.BytesIO(file_bytes))
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        lines.append(("# " + text) if style.startswith(("heading", "title")) else ("• " + text))
    return "\n".join(lines)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right extractor based on the file extension (.pdf or .docx)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == "docx":
        return extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Please upload a .pdf or .docx file.")


# --------------------------------------------------------------------------- text utilities
def clean_text(text: str) -> str:
    """Flatten text: re-join hyphenated words, drop page-number lines, collapse whitespace."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", " ", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Light-weight rule-based sentence splitter (no model download needed)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- structure detection
# bullets / list markers at the start of a line: "•", "-", "o ", "1.", "(1)", "10)"
_BULLET_RE = re.compile(r"^(?:[•\-–*]\s*|o\s+|\(?\d{1,2}[.)]\s*)")
_NUMERIC_MARKER_RE = re.compile(r"^\(?\d")
# labels that are not real section titles
_GENERIC_LABELS = {"example", "examples", "summary", "conclusion", "challenges", "definition",
                   "advantages", "disadvantages", "methods", "common methods", "start", "stop",
                   "output", "key points"}


def _is_heading(text: str, prev: str, nxt: str) -> bool:
    """
    Heuristic: a heading is a short line (<= 9 words) without final punctuation or digits, starting with a
    capital letter, that follows the end of a sentence (or another short line) and is followed by a new
    sentence / list item (not by a lower-case continuation of the same sentence).
    """
    words = text.split()
    if not 1 <= len(words) <= 9 or len(text) < 3:
        return False
    if text[-1] in ".!?,;" or not (text[0].isupper() or text[0] == "("):
        return False
    if re.search(r"[\[\]=0-9]", text):
        return False
    if nxt and nxt[0].islower() and not _BULLET_RE.match(nxt):
        return False
    prev_ok = (not prev) or prev[-1] in ".!?:)" or len(prev.split()) <= 9
    return prev_ok


def parse_units(raw_text: str) -> List[Tuple[str, bool]]:
    """
    Convert raw text (with its line breaks) into a list of (text, is_heading) units.
    Wrapped lines are re-joined; every bullet / numbered item becomes its own unit.
    """
    text = re.sub(r"-\n(?=[a-z])", "", raw_text)
    lines = []
    for line in text.replace("\u00a0", " ").split("\n"):
        line = re.sub(r"[\uf000-\uf8ff\u25cf\u25aa]", "•", line).strip()
        if line and not re.fullmatch(r"\d{1,4}", line):  # drop empty lines and bare page numbers
            lines.append(line)

    units: List[Tuple[str, bool]] = []
    current = None

    def flush():
        nonlocal current
        if current:
            units.append((current, False))
            current = None

    for i, line in enumerate(lines):
        prev = lines[i - 1] if i else ""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        explicit = line.startswith("# ")
        body = line[2:].strip() if explicit else line
        # strip up to two stacked markers, e.g. a glyph bullet followed by "(1)"
        stripped, numeric_marker = body, False
        if not explicit:
            for _ in range(2):
                m = _BULLET_RE.match(stripped)
                if not m:
                    break
                numeric_marker = numeric_marker or bool(_NUMERIC_MARKER_RE.match(stripped))
                stripped = stripped[m.end():].strip()
        is_item = stripped != body
        if not stripped:
            continue
        can_be_heading = explicit or not is_item or numeric_marker
        if can_be_heading and (explicit or _is_heading(stripped, prev, nxt)):
            flush()
            title = stripped.rstrip(":").strip()
            if title.lower() not in _GENERIC_LABELS:
                units.append((title, True))
            continue
        if is_item:
            flush()
            current = stripped
        else:
            current = stripped if current is None else current + " " + stripped
    flush()
    return units


# --------------------------------------------------------------------------- chunking
# sections that list topics / references instead of teaching content
_SKIP_TITLE_RE = re.compile(r"^(syllabus|contents?|table of contents|index|references|bibliography|"
                            r"acknowledg\w*|preface|learning outcomes?)\b", re.I)


@dataclass
class Chunk:
    """A passage that lives inside one section: its title plus the list items / paragraphs it contains."""
    title: str
    units: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.units)


def build_chunks(raw_text: str, max_words: int = 150, split_words: int = 35, min_words: int = 25) -> List[Chunk]:
    """
    Group units into passages of at most `max_words` words that do not cross section boundaries:
    a new heading closes the current passage once it holds at least `split_words` words.
    Passages shorter than `min_words` are discarded.
    """
    chunks: List[Chunk] = []
    current: List[str] = []
    count = 0
    title = ""

    def flush():
        nonlocal current, count
        if current and count >= min_words and not _SKIP_TITLE_RE.match(title):
            chunks.append(Chunk(title, current))
        current, count = [], 0

    for text, is_heading in parse_units(raw_text):
        if is_heading:
            if count >= split_words:
                flush()
            title = text
            continue
        pieces = split_sentences(text) if len(text.split()) > 60 else [text]
        for piece in pieces:
            n_words = len(piece.split())
            if current and count + n_words > max_words:
                flush()
            current.append(piece)
            count += n_words
    flush()
    return chunks
