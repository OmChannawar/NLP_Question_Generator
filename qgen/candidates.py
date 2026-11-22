"""
candidates.py
-------------
Answer-aware question generation, step 1: decide WHAT each question will be about.

Instead of asking the LLM "write any question about this passage" (which makes it invent things), the answer
is picked from the textbook first:

  kind "term"     -> a key TERM (from "Term: definition" / "X is a ..." sentences when possible)
  kind "sentence" -> an informative SENTENCE (mid) or a sentence with reasoning cues (high)

This module also provides the TermPool used for MCQ distractors and True/False statements.
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .extractor import Chunk, split_sentences

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
LABEL_RE = re.compile(r"^(?:definition|example|examples|note)s?:\s*", re.I)
DEF_COLON = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ()/'’\-]{1,45}?):\s+(?P<defn>\S.{14,})$")
DEF_IS = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ()/'’\-]{2,50}?)\s+(?:is|are|refers to)\s+(?:\w+ly\s+)?"
                    r"(?P<rest>(?:a|an|the|one of)\s+\S.+)$")
PRONOUN_STARTS = {"it", "this", "that", "these", "they", "there", "he", "she", "we", "in", "if", "each",
                  "one", "some", "another", "many", "most", "both", "all", "such", "other", "every",
                  "several", "common", "here", "then", "when", "after", "once"}
REASONING_CUES = re.compile(r"\b(because|so that|helps?|prevent\w*|avoid\w*|ensur\w*|allow\w*|enables?|"
                            r"therefore|whereas|while|unlike|advantage\w*|challenge\w*|limitation\w*|"
                            r"trade-?off|balance|difference|compared|instead)\b", re.I)
TERM_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z]*)\b")


@dataclass
class Candidate:
    """One planned question: what it is answered by, where that comes from, and its context."""
    answer: str            # expected answer: a term (kind 'term') or a sentence (kind 'sentence')
    evidence: str          # the textbook sentence that supports the answer
    context: str           # the passage given to the LLM
    title: str             # topic of the passage
    chunk_id: int
    chapter: str = ""
    kind: str = "sentence"
    score: float = 0.0
    defn: str = ""         # for definitions: the description of the term (used by Match the pair)


# --------------------------------------------------------------------------- helpers
def _words(text: str) -> List[str]:
    """Lower-cased content words (stop-words removed)."""
    return [w for w in re.findall(TOKEN_PATTERN, text.lower()) if w not in ENGLISH_STOP_WORDS]


def build_idf(chunks: List[Chunk]) -> Dict[str, float]:
    """Document-level IDF: rare, topic-specific words get high weight, generic words low weight."""
    sentences = [s for c in chunks for u in c.units for s in split_sentences(u)]
    try:
        vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), token_pattern=TOKEN_PATTERN).fit(sentences)
    except ValueError:
        return {}
    return dict(zip(vec.get_feature_names_out(), vec.idf_))


def _info_score(sentence: str, idf: Dict[str, float]) -> float:
    """How information-rich a sentence is (sum of IDF of its distinct content words, length-normalised)."""
    words = set(_words(sentence))
    if not words:
        return 0.0
    default = max(idf.values()) if idf else 1.0
    return sum(idf.get(w, default) for w in words) / math.sqrt(len(words))


def _good_sentence(s: str, min_words: int) -> bool:
    """Reject list junk, formulas, arrays like [1, 0, 1], questions and very short / very long sentences."""
    n = len(s.split())
    if not min_words <= n <= 45 or s.endswith(("?", ":")) or re.search(r"[\[\]=]", s):
        return False
    return sum(c.isalpha() or c.isspace() for c in s) / len(s) >= 0.85


def _strip_article(term: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", term.strip(), flags=re.I)


def _definition(sentence: str):
    """Return (term, answer_sentence, description) if the sentence defines a term, else (None, sentence, '')."""
    m = DEF_COLON.match(sentence)
    if m:
        term = m.group("term").strip()
        if len(term.split()) <= 6 and _words(term):
            return _strip_article(term), m.group("defn").strip(), m.group("defn").strip()
    m = DEF_IS.match(sentence)
    if m:
        term = m.group("term").strip()
        first = term.split()[0].lower()
        if (len(term.split()) <= 6 and _words(term) and first not in PRONOUN_STARTS
                and len(m.group("rest").split()) >= 4):
            return _strip_article(term), sentence, m.group("rest").strip()
    return None, sentence, ""


def _key_term(sentence: str) -> Optional[str]:
    """Fallback term: the longest capitalised phrase / acronym (no guessing otherwise)."""
    found = [t for t in TERM_RE.findall(sentence) if _words(t)]
    return max(found, key=len) if found else None


# --------------------------------------------------------------------------- candidate extraction
def chunk_candidates(chunk: Chunk, difficulty: str, idf: Dict[str, float], kind: Optional[str] = None) -> List[Candidate]:
    """All possible questions for one passage, best first. kind: 'term' or 'sentence' (default by difficulty)."""
    kind = kind or ("term" if difficulty == "low" else "sentence")
    candidates, seen = [], set()
    min_words = 6 if kind == "term" else 8
    for unit in chunk.units:
        unit = LABEL_RE.sub("", unit)
        for sentence in split_sentences(unit):
            if not _good_sentence(sentence, min_words) or sentence in seen:
                continue
            seen.add(sentence)
            term, full, defn = _definition(sentence)
            score = _info_score(sentence, idf)
            if sentence.split()[0].lower().strip(",") in PRONOUN_STARTS:
                score *= 0.2  # starts with "This / It / These ..." -> not understandable on its own
            if kind == "term":
                answer = term or _key_term(sentence)
                if not answer:
                    continue
                score *= 1.5 if term else 0.6
            else:
                answer = full
                if difficulty == "high":
                    score *= 2.5 if REASONING_CUES.search(sentence) else 0.6
                else:
                    score *= 1.2 if term else 1.0
            candidates.append(Candidate(answer, sentence, chunk.text, chunk.title, chunk.id,
                                        chunk.chapter, kind, score, defn))
    return sorted(candidates, key=lambda c: -c.score)


def plan_slots(chunks: List[Chunk], n: int, difficulty: str, kind: Optional[str] = None,
               exclude: Optional[Set[str]] = None, idf: Optional[Dict[str, float]] = None,
               cache: Optional[dict] = None, max_alternates: int = 2) -> List[List[Candidate]]:
    """
    Choose `n` questions spread over `chunks`. Returns one *slot* per question; a slot is a list of candidates
    (best first): the first is tried first, the others are fall-backs from the same passage.
    `exclude` holds evidence sentences already used by other question types; `cache` shares computed candidates.
    Fewer than `n` slots are returned if the text has too few suitable sentences.
    """
    kind = kind or ("term" if difficulty == "low" else "sentence")
    exclude = exclude or set()
    idf = idf if idf is not None else build_idf(chunks)
    store = cache.setdefault((kind, difficulty if kind == "sentence" else "-"), {}) if cache is not None else {}
    per_chunk = []
    for c in chunks:
        if c.id not in store:
            store[c.id] = chunk_candidates(c, difficulty, idf, kind)
        per_chunk.append([x for x in store[c.id] if x.evidence not in exclude])
    usable = [i for i, c in enumerate(per_chunk) if c]
    if not usable:
        return []

    picks = []  # (index in `chunks`, rank)
    if len(usable) >= n:
        # keep the passages with the strongest best-candidate (drops weak ones), then space them evenly
        keep = sorted(usable, key=lambda i: -per_chunk[i][0].score)[:max(n, int(len(usable) * 0.7))]
        keep.sort()
        for j in np.linspace(0, len(keep) - 1, n).round().astype(int):
            picks.append((keep[j], 0))
    else:  # fewer passages than questions -> 2nd, 3rd ... best sentences round-robin
        rank = 0
        while len(picks) < n:
            added = False
            for i in usable:
                if len(per_chunk[i]) > rank and len(picks) < n:
                    picks.append((i, rank))
                    added = True
            if not added:
                break
            rank += 1
    picks.sort()

    used = {(i, r) for i, r in picks}
    slots = []
    for i, r in picks:
        alternates = [per_chunk[i][k] for k in range(len(per_chunk[i])) if (i, k) not in used][:max_alternates]
        slots.append([per_chunk[i][r]] + alternates)
    return slots


# --------------------------------------------------------------------------- term pool (distractors)
@dataclass
class TermEntry:
    """A term found in the book, with the sentence it appears in and where."""
    term: str
    evidence: str
    chunk_id: int
    chapter: str = ""
    topic: str = ""
    defn: str = ""


class TermPool:
    """All distinct terms of the book. Provides plausible WRONG options for MCQ and True/False questions."""

    def __init__(self, entries: List[TermEntry]):
        seen, self.entries = set(), []
        for e in entries:
            key = e.term.lower()
            if key not in seen:
                seen.add(key)
                self.entries.append(e)
        try:
            self._vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), token_pattern=TOKEN_PATTERN)
            self._mat = self._vec.fit_transform([e.evidence for e in self.entries])
        except ValueError:
            self._vec, self._mat = None, None

    def similarity(self, text: str) -> np.ndarray:
        """Cosine similarity of `text` to the evidence sentence of every entry."""
        if self._mat is None:
            return np.zeros(len(self.entries))
        return (self._vec.transform([text]) @ self._mat.T).toarray()[0]

    def distractors(self, answer: str, chunk_id: int, chapter: str, evidence: str, difficulty: str,
                    k: int, rng, forbid_text: str = "") -> List[str]:
        """
        Pick `k` wrong options. Difficulty controls how plausible they are:
          low  -> terms from far-away parts of the book (easy to rule out)
          mid  -> terms from the same chapter
          high -> terms whose sentences are most similar to the correct one (hard to rule out)
        """
        a, forbid = answer.lower(), forbid_text.lower()
        ok = []
        for i, e in enumerate(self.entries):
            t = e.term.lower()
            if t == a or a in t or t in a or t in forbid or abs(len(e.term.split()) - len(answer.split())) > 2:
                continue
            ok.append(i)
        if difficulty == "high":
            sims = self.similarity(evidence)
            ranked = sorted(ok, key=lambda i: -sims[i])
        elif difficulty == "mid":
            same = [i for i in ok if self.entries[i].chapter == chapter and self.entries[i].chunk_id != chunk_id]
            rest = [i for i in ok if i not in set(same)]
            rng.shuffle(same)
            rng.shuffle(rest)
            ranked = same + rest
        else:
            far = sorted(ok, key=lambda i: -abs(self.entries[i].chunk_id - chunk_id))[:max(k * 4, len(ok) // 2)]
            rng.shuffle(far)
            ranked = far
        chosen, seen = [], set()
        for i in ranked:
            t = self.entries[i].term
            if t.lower() not in seen:
                seen.add(t.lower())
                chosen.append(t)
            if len(chosen) == k:
                break
        return chosen
