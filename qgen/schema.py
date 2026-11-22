"""
schema.py
---------
Shared definitions: question types, difficulty levels, the QAItem data class and request validation.
(Kept free of heavy imports such as torch so that every module can use it.)
"""
from dataclasses import dataclass, field
from typing import Dict, List

DIFFICULTY_LEVELS = ["low", "mid", "high"]
ALLOWED_COUNTS = [5, 10, 15, 20, 25, 30]  # multiples of 5, max 30

# order matters: it is the order in which types are generated and listed
QUESTION_TYPES = {
    "subjective": "Subjective (written answer)",
    "mcq": "Multiple choice (MCQ)",
    "fill_blank": "Fill in the blanks",
    "true_false": "True / False",
    "match": "Match the pair",
}
LLM_TYPES = {"subjective", "mcq"}  # types whose question text is written by the Seq2Seq LLM


@dataclass
class QAItem:
    """One generated quiz entry (all question types share this structure)."""
    question: str                                   # question text / statement / sentence with blanks
    answer: str                                     # expected answer (term, sentence, 'True', 'B' ...)
    keywords: List[str] = field(default_factory=list)
    evidence: str = ""                              # textbook sentence the answer comes from
    qtype: str = "subjective"
    chapter: str = ""
    topic: str = ""
    chunk_id: int = 0                               # position in the book (used for document order)
    options: List[str] = field(default_factory=list)      # MCQ options, or the right column of a match set
    correct_index: int = -1                         # MCQ: index of the correct option
    left: List[str] = field(default_factory=list)         # match: left column (terms)
    match_key: List[int] = field(default_factory=list)    # match: index in `options` for each left item
    uid: int = 0                                    # 0-based number after sorting in document order


def validate_request(num_questions: int, difficulty: str) -> None:
    """Project rules: difficulty in {low, mid, high}; count is a multiple of 5 between 5 and 30."""
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"Difficulty must be one of {DIFFICULTY_LEVELS}.")
    if num_questions % 5 != 0 or not 5 <= num_questions <= 30:
        raise ValueError("Number of questions must be a multiple of 5 between 5 and 30.")


def allocate(num_questions: int, qtypes: List[str]) -> Dict[str, int]:
    """Split `num_questions` as evenly as possible over the chosen types (earlier types get the remainder)."""
    types = [t for t in QUESTION_TYPES if t in qtypes]
    if not types:
        raise ValueError("Choose at least one question type.")
    base, extra = divmod(num_questions, len(types))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(types)}
