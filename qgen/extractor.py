"""
extractor.py
------------
Document ingestion for whole textbooks:
  * reads PDF / DOCX (page by page, with progress),
  * removes running headers / footers / page numbers,
  * detects CHAPTERS and TOPICS (headings),
  * builds section-aware passages that never cross a topic boundary.

Every passage carries its chapter and topic title, which are used to sort and filter the questions later.
"""
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from .progress import ProgressTracker, tick

STAGE_READ = "Reading document"
STAGE_CLEAN = "Cleaning headers & footers"
STAGE_STRUCT = "Detecting chapters & topics"
ANALYSIS_STAGES = [(STAGE_READ, 6.0), (STAGE_CLEAN, 0.5), (STAGE_STRUCT, 2.0)]


# --------------------------------------------------------------------------- file reading
def _strip_headers_footers(pages: List[List[str]]) -> List[List[str]]:
    """
    Remove running headers / footers: a line that is one of the first or last two lines on at least 30 % of
    the pages (digits ignored, so 'Page 12' and 'Page 13' count as the same line) is deleted everywhere.
    """
    n = len(pages)
    if n < 8:
        return pages

    def edges(lines):
        nonempty = [i for i, l in enumerate(lines) if l.strip()]
        return set(nonempty[:2] + nonempty[-2:])

    def norm(s):
        return re.sub(r"\d+", "#", s.strip().lower())

    counts: Counter = Counter()
    for lines in pages:
        for key in {norm(lines[i]) for i in edges(lines)}:
            counts[key] += 1
    threshold = max(4, int(0.3 * n))
    bad = {k for k, c in counts.items() if c >= threshold and k}
    return [[l for i, l in enumerate(lines) if not (i in edges(lines) and norm(l) in bad)] for lines in pages]


def extract_text_from_pdf(file_bytes: bytes, progress: Optional[ProgressTracker] = None) -> str:
    """Return the text of every page of a PDF (one text line per line); pages that fail are skipped."""
    reader = PdfReader(io.BytesIO(file_bytes))
    total = len(reader.pages)
    pages: List[List[str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # corrupt / image-only page
            text = ""
        pages.append(text.split("\n"))
        if i % 5 == 0 or i == total - 1:
            tick(progress, STAGE_READ, (i + 1) / total, f"page {i + 1} of {total}")
    tick(progress, STAGE_CLEAN, 0.0, "looking for repeated headers and footers")
    pages = _strip_headers_footers(pages)
    tick(progress, STAGE_CLEAN, 1.0, "done")
    return "\n".join("\n".join(lines) for lines in pages)


def extract_text_from_docx(file_bytes: bytes, progress: Optional[ProgressTracker] = None) -> str:
    """
    Return the text of a Word document, one paragraph per line. Heading styles are kept as markers
    ('§H1§ ' = Heading 1 ...); other paragraphs get a bullet so each paragraph stays a separate unit.
    """
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = doc.paragraphs
    lines = []
    for i, p in enumerate(paragraphs):
        text = p.text.strip()
        if text:
            style = (p.style.name or "").lower() if p.style is not None else ""
            level = re.search(r"heading\s*(\d)", style)
            if style.startswith("title"):
                lines.append(f"§H1§ {text}")
            elif level:
                lines.append(f"§H{min(int(level.group(1)), 6)}§ {text}")
            else:
                lines.append("• " + text)
        if i % 200 == 0 or i == len(paragraphs) - 1:
            tick(progress, STAGE_READ, (i + 1) / max(len(paragraphs), 1), f"paragraph {i + 1} of {len(paragraphs)}")
    tick(progress, STAGE_CLEAN, 1.0, "nothing to clean in a Word file")
    return "\n".join(lines)


def extract_text(filename: str, file_bytes: bytes, progress: Optional[ProgressTracker] = None) -> str:
    """Dispatch on the file extension (.pdf or .docx)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes, progress)
    if ext == "docx":
        return extract_text_from_docx(file_bytes, progress)
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
_BULLET_RE = re.compile(r"^(?:[•\-–*]\s*|o\s+|\(?\d{1,2}[.)]\s*)")
_NUMERIC_MARKER_RE = re.compile(r"^\(?\d")
_EXPLICIT_RE = re.compile(r"^§H([1-6])§ (.*)$")
# "Chapter 3", "Chapter 3: Genetic Algorithms", "Unit IV - Search", "Module 2 Optimisation"
_CHAPTER_RE = re.compile(r"^(?i:chapter|unit|module|part|lesson)\s+(?:\d{1,3}|[ivxlc]{1,6})\b"
                         r"(?:\s*[:.\-–—]\s*(?P<rest>.*)|\s+(?P<rest2>[A-Z].*)|\s*)$")
# "3.1 Selection methods", "2.4.1 Roulette wheel"
_NUM_HEAD_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\.?\s+[A-Z][^.!?]{2,90}$")
_GENERIC_LABELS = {"example", "examples", "summary", "conclusion", "challenges", "definition",
                   "advantages", "disadvantages", "methods", "common methods", "start", "stop",
                   "output", "key points"}
_SKIP_TITLE_RE = re.compile(r"^(syllabus|contents?|table of contents|index|references|bibliography|"
                            r"acknowledg\w*|preface|learning outcomes?)\b", re.I)


_LEAD_IN_RE = re.compile(r"^(here|let|imagine|consider|note|suppose|assume|following)\b", re.I)


def _clean_title(title: str) -> str:
    """'Key characteristics of swarm intelligence include' -> 'Key characteristics of swarm intelligence'."""
    title = re.sub(r"\s+(?:include|includes|are|is|as follows)$", "", title.strip().rstrip(":"), flags=re.I)
    return title.strip()


def _is_heading(text: str, prev: str, nxt: str) -> bool:
    """Short line (<= 9 words), no final punctuation/digits, capitalised, followed by a new sentence or item."""
    words = text.split()
    if not 1 <= len(words) <= 9 or len(text) < 3:
        return False
    if text[-1] in ".!?,;" or not (text[0].isupper() or text[0] == "("):
        return False
    if _LEAD_IN_RE.match(text):          # "Here's how it works:", "Consider the following:" are not topics
        return False
    if re.search(r"[\[\]=0-9]", text):
        return False
    if nxt and nxt[0].islower() and not _BULLET_RE.match(nxt):
        return False
    return (not prev) or prev[-1] in ".!?:)" or len(prev.split()) <= 9


def _looks_like_title(line: str) -> bool:
    """Used for 'Chapter 3' followed by its title on the next line."""
    words = line.split()
    return (1 <= len(words) <= 9 and line[0].isupper() and line[-1] not in ".!?,;"
            and not _BULLET_RE.match(line))


def parse_units(raw_text: str) -> List[Tuple[str, str]]:
    """
    Convert raw text (with line breaks) into (text, kind) units, kind in {'text', 'topic', 'chapter'}.
    Wrapped lines are re-joined; every bullet / numbered item is its own unit.
    """
    text = re.sub(r"-\n(?=[a-z])", "", raw_text)
    lines = []
    for line in text.replace("\u00a0", " ").split("\n"):
        line = re.sub(r"[\uf000-\uf8ff\u25cf\u25aa]", "•", line).strip()
        if line and not re.fullmatch(r"\d{1,4}", line):  # drop empty lines and bare page numbers
            lines.append(line)
    has_deeper = any(re.match(r"§H[2-6]§ ", l) for l in lines)

    units: List[Tuple[str, str]] = []
    current: Optional[str] = None

    def flush():
        nonlocal current
        if current:
            units.append((current, "text"))
            current = None

    i = 0
    while i < len(lines):
        line = lines[i]
        prev = lines[i - 1] if i else ""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        m = _EXPLICIT_RE.match(line)
        level = int(m.group(1)) if m else 0
        body = m.group(2).strip() if m else line
        skip = 0

        # 1) chapter / numbered-section headings, 2) Word heading styles
        kind, title = None, ""
        if len(body.split()) <= 12 and not body.endswith((".", "!", "?", ",", ";")):
            if _CHAPTER_RE.match(body):
                kind, title = "chapter", body.rstrip(":").strip()
                if re.fullmatch(r"(?i:chapter|unit|module|part|lesson)\s+\S+", body) and nxt and _looks_like_title(nxt):
                    title, skip = f"{title}: {nxt}", 1          # 'Chapter 3' + title on the next line
            elif not level and _NUM_HEAD_RE.match(body):
                kind, title = "topic", body
        if kind is None and level:
            kind = "chapter" if (level == 1 and has_deeper) else "topic"
            title = body.rstrip(":").strip()
        if kind:
            flush()
            if not (kind == "topic" and title.lower() in _GENERIC_LABELS):
                units.append((title, kind))
            i += 1 + skip
            continue

        # 3) heuristic headings and list items (as before)
        stripped, numeric_marker = body, False
        for _ in range(2):
            mm = _BULLET_RE.match(stripped)
            if not mm:
                break
            numeric_marker = numeric_marker or bool(_NUMERIC_MARKER_RE.match(stripped))
            stripped = stripped[mm.end():].strip()
        is_item = stripped != body
        i += 1
        if not stripped:
            continue
        if (not is_item or numeric_marker) and _is_heading(stripped, prev, nxt):
            flush()
            title = _clean_title(stripped)
            if title.lower() not in _GENERIC_LABELS:
                units.append((title, "topic"))
            continue
        if is_item:
            flush()
            current = stripped
        else:
            current = stripped if current is None else current + " " + stripped
    flush()
    return units


# --------------------------------------------------------------------------- chunking
@dataclass
class Chunk:
    """A passage inside one topic: its chapter, topic title, position and the items/paragraphs it contains."""
    title: str                                   # topic
    units: List[str] = field(default_factory=list)
    chapter: str = ""
    id: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.units)

    @property
    def topic_key(self) -> str:
        """Unique label of the chapter/topic pair, used for filtering."""
        return f"{self.chapter} › {self.title}" if self.chapter else self.title


def _label_untitled(chunks: List[Chunk]) -> None:
    """Documents without headings: group neighbouring passages and name each group by its top TF-IDF words."""
    idx = [i for i, c in enumerate(chunks) if not c.title]
    if not idx:
        return
    groups, cur = [], []
    for i in idx:
        if cur and (i != cur[-1] + 1 or chunks[i].chapter != chunks[cur[0]].chapter or len(cur) >= 4):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    texts = [" ".join(chunks[i].text for i in g) for g in groups]
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{3,}\b",
                              max_df=0.8 if len(texts) > 4 else 1.0)
        matrix = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()
    except ValueError:
        matrix, terms = None, []
    for n, g in enumerate(groups):
        label = ""
        if matrix is not None:
            row = matrix[n].toarray()[0]
            label = ", ".join(terms[j] for j in row.argsort()[::-1][:3] if row[j] > 0)
        for i in g:
            chunks[i].title = f"Topic: {label}" if label else f"Part {n + 1}"


def build_chunks(raw_text: str, max_words: int = 150, split_words: int = 35, min_words: int = 25,
                 progress: Optional[ProgressTracker] = None) -> List[Chunk]:
    """
    Group units into passages of at most `max_words` words that never cross a topic or chapter boundary
    (a heading closes the current passage once it holds at least `split_words` words; a chapter heading
    always does). Passages shorter than `min_words` and non-content sections (syllabus, index ...) are dropped.
    """
    units = parse_units(raw_text)
    chunks: List[Chunk] = []
    current: List[str] = []
    count, chapter, title = 0, "", ""

    def flush():
        nonlocal current, count
        if current and count >= min_words and not _SKIP_TITLE_RE.match(title) and not _SKIP_TITLE_RE.match(chapter):
            chunks.append(Chunk(title, current, chapter))
        current, count = [], 0

    for k, (text, kind) in enumerate(units):
        if k % 200 == 0:
            tick(progress, STAGE_STRUCT, 0.9 * k / max(len(units), 1), f"section {k} of {len(units)}")
        if kind == "chapter":
            flush()
            chapter, title = text, text
            continue
        if kind == "topic":
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
    tick(progress, STAGE_STRUCT, 0.95, "naming sections")
    _label_untitled(chunks)
    for i, c in enumerate(chunks):
        c.id = i
    tick(progress, STAGE_STRUCT, 1.0, f"{len(chunks)} passages")
    return chunks


def outline(chunks: List[Chunk]) -> Dict[str, Dict[str, int]]:
    """chapter -> {topic -> number of passages}, in book order ('' is used when a book has no chapters)."""
    result: Dict[str, Dict[str, int]] = {}
    for c in chunks:
        result.setdefault(c.chapter, {})
        result[c.chapter][c.title] = result[c.chapter].get(c.title, 0) + 1
    return result
