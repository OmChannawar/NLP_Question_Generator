"""
qtypes.py
---------
Builders for the question types that do NOT need the language model to write the question text:

  Fill in the blanks -> a textbook sentence with its key term(s) blanked out
  True / False       -> a textbook sentence, either unchanged (True) or with its key term swapped for a
                        different term from the book (False)
  Match the pair     -> 4 terms and their descriptions taken from definitions in the book
  MCQ options        -> wrong answers (distractors) for the questions the LLM wrote for a term

Difficulty is expressed through the choices: more blanks, or more plausible wrong options / pairs.
"""
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .candidates import TOKEN_PATTERN, Candidate, TermEntry, TermPool, _words
from .keywords import extract_keywords
from .schema import QAItem


def _pattern(text: str):
    """Case-insensitive whole-word pattern for a term (works for terms containing brackets)."""
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.I)


def _find_term(sentence: str, term: str):
    """Return a compiled pattern that matches `term` in `sentence` (also without a trailing '(ABC)'), or None."""
    pat = _pattern(term)
    if pat.search(sentence):
        return pat
    base = re.sub(r"\s*\(.*?\)", "", term).strip()
    if base:
        pat = _pattern(base)
        if pat.search(sentence):
            return pat
    return None


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- fill in the blanks
def _blank_item(cand: Candidate, difficulty: str, idf) -> "QAItem | None":
    """Blank out the candidate's term (and, for high difficulty, a second key word) in its sentence."""
    sentence, term = cand.evidence, cand.answer
    if len(sentence.split()) > 35 or (difficulty == "low" and len(sentence.split()) > 28):
        return None
    pat = _find_term(sentence, term)
    if pat is None:
        return None
    matches = [m.group(0) for m in pat.finditer(sentence)]
    if len(matches) > 2:
        return None
    blanked = pat.sub("_____", sentence)
    answers = [matches[0]]
    if difficulty == "high":  # second blank: the rarest remaining content word
        words = [w for w in re.findall(TOKEN_PATTERN, blanked)
                 if w.lower() not in ENGLISH_STOP_WORDS and len(w) >= 6 and w.lower() not in term.lower()]
        if words:
            second = max(set(words), key=lambda w: idf.get(w.lower(), 0.0))
            blanked = _pattern(second).sub("_____", blanked, count=1)
            answers.append(second)
    if len(blanked.replace("_____", "").split()) < 6:
        return None
    joined = " ".join(answers).lower()
    keywords = [a.lower() for a in answers] + [k for k in extract_keywords(sentence, cand.context, 4) if k not in joined]
    return QAItem(blanked, "; ".join(answers), keywords[:6], sentence, "fill_blank",
                  cand.chapter, cand.title, cand.chunk_id)


def build_fill_blank(slots: List[List[Candidate]], difficulty: str, idf) -> List[QAItem]:
    """One fill-in-the-blank question per slot (falls back to the slot's alternative sentences)."""
    items = []
    for slot in slots:
        for cand in slot:
            item = _blank_item(cand, difficulty, idf)
            if item:
                items.append(item)
                break
    return items


# --------------------------------------------------------------------------- true / false
def _tf_item(cand: Candidate, make_false: bool, difficulty: str, pool: TermPool, rng) -> "QAItem | None":
    """Build a True statement (the sentence itself) or a False one (key term replaced by another term)."""
    statement = cand.evidence
    if not 6 <= len(statement.split()) <= 35:
        return None
    keywords = [cand.answer.lower()] + [k for k in extract_keywords(statement, cand.context, 4)
                                        if k not in cand.answer.lower()]
    if not make_false:
        return QAItem(statement, "True", keywords[:5], statement, "true_false", cand.chapter, cand.title, cand.chunk_id)
    pat = _find_term(statement, cand.answer)
    wrong = pool.distractors(cand.answer, cand.chunk_id, cand.chapter, cand.evidence, difficulty, 1, rng,
                             forbid_text=statement)
    if pat is None or not wrong:
        return None
    swapped = pat.sub(lambda m: _cap(wrong[0]) if m.start() == 0 else wrong[0], statement, count=1)
    return QAItem(swapped, "False", keywords[:5], statement, "true_false", cand.chapter, cand.title, cand.chunk_id)


def build_true_false(slots: List[List[Candidate]], difficulty: str, pool: TermPool, rng) -> List[QAItem]:
    """Alternate True / False statements (random start); falls back to a True statement if no False one fits."""
    items, want_false = [], rng.random() < 0.5
    for slot in slots:
        made = None
        for polarity in ([want_false, False] if want_false else [False]):
            for cand in slot:
                made = _tf_item(cand, polarity, difficulty, pool, rng)
                if made:
                    break
            if made:
                break
        if made:
            items.append(made)
            want_false = made.answer == "True"
    return items


# --------------------------------------------------------------------------- MCQ
def add_mcq_options(items: List[QAItem], pool: TermPool, difficulty: str, rng) -> List[QAItem]:
    """Attach 3 distractors to LLM-written term questions; questions that cannot get 3 are dropped."""
    kept = []
    for it in items:
        wrong = pool.distractors(it.answer, it.chunk_id, it.chapter, it.evidence, difficulty, 3, rng,
                                 forbid_text=it.question)
        if len(wrong) < 3:
            continue
        it.answer = _cap(it.answer)
        options = [it.answer] + [_cap(w) for w in wrong]
        rng.shuffle(options)
        it.options, it.correct_index, it.qtype = options, options.index(it.answer), "mcq"
        kept.append(it)
    return kept


# --------------------------------------------------------------------------- match the pair
def _defn_ok(e: TermEntry) -> bool:
    """A usable definition: 4-28 words and it does not contain the words of its own term."""
    n = len(e.defn.split())
    return 4 <= n <= 28 and not (set(_words(e.term)) & set(_words(e.defn)))


def _compatible(a: TermEntry, b: TermEntry) -> bool:
    """Two entries can sit in one match set if neither description mentions the other's term and the terms differ."""
    if set(_words(a.term)) & set(_words(b.term)):
        return False
    return not (_pattern(b.term).search(a.defn) or _pattern(a.term).search(b.defn))


def build_match(definitions: List[TermEntry], n_sets: int, difficulty: str, rng, size: int = 4) -> List[QAItem]:
    """
    Build up to `n_sets` match-the-pair questions of `size` terms each.
      low  -> terms from far-apart parts of the book
      mid  -> terms from the same chapter
      high -> terms whose descriptions are most similar to each other
    """
    clean = [e for e in definitions if _defn_ok(e)]
    if len(clean) < size or n_sets < 1:
        return []
    pool = TermPool(clean)
    entries = pool.entries
    used, items = set(), []
    for s in [int(round(x)) for x in np.linspace(0, len(entries) - 1, n_sets)]:
        free = [i for i in range(len(entries)) if i not in used]
        if len(free) < size:
            break
        seed = min(free, key=lambda i: abs(i - s))
        others = [i for i in free if i != seed]
        if difficulty == "high":
            sims = pool.similarity(entries[seed].evidence)
            ranked = sorted(others, key=lambda i: -sims[i])
        elif difficulty == "mid":
            same = [i for i in others if entries[i].chapter == entries[seed].chapter]
            rest = [i for i in others if i not in set(same)]
            rng.shuffle(same)
            rng.shuffle(rest)
            ranked = same + rest
        else:
            far = sorted(others, key=lambda i: -abs(entries[i].chunk_id - entries[seed].chunk_id))
            far = far[:max(size * 4, len(far) // 2)]
            rng.shuffle(far)
            ranked = far
        group = [seed]
        for i in ranked:
            if all(_compatible(entries[i], entries[j]) for j in group):
                group.append(i)
            if len(group) == size:
                break
        if len(group) < size:
            continue
        used.update(group)
        group.sort(key=lambda i: entries[i].chunk_id)
        left = [_cap(entries[i].term) for i in group]
        descriptions = [_cap(entries[i].defn if entries[i].defn.endswith(".") else entries[i].defn + ".") for i in group]
        order = list(range(size))
        rng.shuffle(order)                                   # order[k] = which description sits in right slot k
        right = [descriptions[j] for j in order]
        key = [order.index(i) for i in range(size)]          # right slot of each left term
        letters = "ABCDEFGH"
        answer = ", ".join(f"{i + 1}-{letters[key[i]]}" for i in range(size))
        seed_e = entries[seed]
        items.append(QAItem("Match the following terms with their descriptions.", answer,
                            [t.lower() for t in left], seed_e.evidence, "match", seed_e.chapter, seed_e.topic,
                            seed_e.chunk_id, options=right, left=left, match_key=key))
    return items
