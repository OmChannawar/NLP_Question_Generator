"""
pipeline.py
-----------
End-to-end orchestration:  text -> clean -> chunk -> select -> generate (Seq2Seq LLM) -> keywords.
"""
from typing import Callable, List, Optional

from .extractor import build_chunks, clean_text, select_chunks
from .generator import QAItem, QuestionGenerator, validate_request


def generate_quiz(text: str, generator: QuestionGenerator, difficulty: str, num_questions: int,
                  progress_callback: Optional[Callable[[float], None]] = None) -> List[QAItem]:
    """
    Generate up to `num_questions` questions spread across the book.

    progress_callback receives a float in [0, 1] after each passage (used by the UI progress bar).
    May return fewer questions than requested if the text is too short to yield unique questions.
    """
    validate_request(num_questions, difficulty)
    chunks = build_chunks(clean_text(text))
    if not chunks:
        raise ValueError("No readable prose found. The file may be scanned images or too short.")

    items: List[QAItem] = []
    for i, context in enumerate(select_chunks(chunks, num_questions), start=1):
        item = generator.generate_item(context, difficulty, [it.question for it in items])
        if item:
            items.append(item)
        if progress_callback:
            progress_callback(i / num_questions)
    return items
