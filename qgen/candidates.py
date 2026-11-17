"""
candidates.py
-------------
Answer-aware question generation, step 1: decide WHAT each question will be about.

Instead of asking the LLM "write any question about this passage" (which makes it invent things), we first
pick an answer that really exists in the textbook, then ask the LLM to write a question for that answer.

  low  -> answer is a key TERM (taken from "Term: definition" / "X is a ..." sentences when possible)
  mid  -> answer is a whole informative SENTENCE (explain / describe style question)
  high -> answer is a SENTENCE that contains reasoning cues (because, helps, prevents, advantage ...)
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .extractor import Chunk, split_sentences

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
LABEL_RE = re.compile(r"^(?:definition|example|examples|note)s?:\s*", re.I)
DEF_COLON = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ()/'’\-]{1,45}?):\s+(?P<defn>\S.{14,})$")
DEF_IS = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ()/'’\-]{2,50}?)\s+(?:is|are|refers to)\s+(?:\w+ly\s+)?"
                    r"(?:a|an|the|one of)\s+(?P<rest>\S.+)$")
PRONOUN_STARTS = {"it", "this", "that", "these", "they", "there", "he", "she", "we", "in", "if", "each",
                  "one", "some", "another", "many", "most", "both", "all", "such", "other", "every",
                  "several", "common", "here", "then", "when", "after", "once"}
REASONING_CUES = re.compile(r"\b(because|so that|helps?|prevent\w*|avoid\w*|ensur\w*|allow\w*|enables?|"
                            r"therefore|whereas|while|unlike|advantage\w*|challenge\w*|limitation\w*|"
                            r"trade-?off|balance|difference|compared|instead)\b", re.I)
TERM_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z]*)\b")


@dataclass
class Candidate:
    """One planned question: what it should be answered by, where that comes from, and its context."""
    answer: str      # expected answer: a term (low) or a sentence (mid/high)
    evidence: str    # the textbook sentence that supports the answer
    context: str     # the passage given to the LLM
    title: str       # section title of the passage
    chunk_id: int
    score: float = 0.0


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
    if not min_words <= n <= 45 or s.endswith("?") or re.search(r"[\[\]=]", s):
        return False
    return sum(c.isalpha() or c.isspace() for c in s) / len(s) >= 0.85


def _strip_article(term: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", term.strip(), flags=re.I)


def _definition(sentence: str):
    """Return (term, full_sentence_answer) if the sentence defines a term, else (None, sentence)."""
    m = DEF_COLON.match(sentence)
    if m:
        term = m.group("term").strip()
        if len(term.split()) <= 6 and _words(term):
            return _strip_article(term), m.group("defn").strip()
    m = DEF_IS.match(sentence)
    if m:
        term = m.group("term").strip()
        first = term.split()[0].lower()
        if (len(term.split()) <= 6 and _words(term) and first not in PRONOUN_STARTS
                and len(m.group("rest").split()) >= 4):
            return _strip_article(term), sentence
    return None, sentence


def _key_term(sentence: str) -> Optional[str]:
    """Fallback term for low difficulty: the longest capitalised phrase / acronym (no guessing otherwise)."""
    found = [t for t in TERM_RE.findall(sentence) if _words(t)]
    return max(found, key=len) if found else None


# --------------------------------------------------------------------------- candidate extraction
def chunk_candidates(chunk: Chunk, chunk_id: int, difficulty: str, idf: Dict[str, float]) -> List[Candidate]:
    """All possible questions for one passage, best first."""
    candidates, seen = [], set()
    min_words = 6 if difficulty == "low" else 8
    for unit in chunk.units:
        unit = LABEL_RE.sub("", unit)
        for sentence in split_sentences(unit):
            if not _good_sentence(sentence, min_words) or sentence in seen:
                continue
            seen.add(sentence)
            term, full = _definition(sentence)
            score = _info_score(sentence, idf)
            if sentence.split()[0].lower().strip(",") in PRONOUN_STARTS:
                score *= 0.2  # starts with "This / It / These ..." -> not understandable on its own
            if difficulty == "low":
                answer = term or _key_term(sentence)
                if not answer:
                    continue
                score *= 1.5 if term else 0.6
            elif difficulty == "mid":
                answer = full
                score *= 1.2 if term else 1.0
            else:  # high
                answer = full
                score *= 2.5 if REASONING_CUES.search(sentence) else 0.6
            candidates.append(Candidate(answer, sentence, chunk.text, chunk.title, chunk_id, score))
    return sorted(candidates, key=lambda c: -c.score)


def plan_slots(chunks: List[Chunk], n: int, difficulty: str, max_alternates: int = 2) -> List[List[Candidate]]:
    """
    Choose `n` questions spread across the document. Returns one *slot* per question; a slot is a list of
    candidates (best first): the first is tried first, the others are fall-backs from the same passage
    if the generated question is rejected.  Fewer than `n` slots are returned if the text has too few
    suitable sentences.
    """
    idf = build_idf(chunks)
    per_chunk = [chunk_candidates(c, i, difficulty, idf) for i, c in enumerate(chunks)]
    usable = [i for i, c in enumerate(per_chunk) if c]
    if not usable:
        return []

    picks = []  # (chunk_id, rank)
    if len(usable) >= n:  # more passages than questions
        # keep the passages with the strongest best-candidate (drops weak ones), then space them evenly
        keep = sorted(usable, key=lambda i: -per_chunk[i][0].score)[:max(n, int(len(usable) * 0.7))]
        keep.sort()
        for j in np.linspace(0, len(keep) - 1, n).round().astype(int):
            picks.append((keep[j], 0))
    else:  # fewer passages than questions -> take 2nd, 3rd ... best sentences round-robin
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
