"""
pipeline.py
-----------
End-to-end orchestration:  text -> clean -> chunk -> select -> generate (Seq2Seq LLM) -> keywords.
"""
from typing import Callable, List, Optional

from .extractor import build_chunks, clean_text, select_chunks
from .generator import QAItem, QuestionGenerator, validate_request


def generate_quiz(text: str, generator: QuestionGenerator, difficulty: str, num_questions: int,
                  progress_callback: Optional[Callable[[float], None]] = None,
                  num_beams: int = 1) -> List[QAItem]:
    """
    Generate up to `num_questions` questions spread across the book.

    num_beams=1 -> fast greedy decoding (default); num_beams=4 -> beam search (better quality, slower).
    progress_callback receives a float in [0, 1] (used by the UI progress bar).
    May return fewer questions than requested if the text is too short to yield unique questions.
    """
    validate_request(num_questions, difficulty)
    chunks = build_chunks(clean_text(text))
    if not chunks:
        raise ValueError("No readable prose found. The file may be scanned images or too short.")

    targets = select_chunks(chunks, num_questions)
    return generator.generate_items(targets, difficulty, num_beams=num_beams,
                                    progress_callback=progress_callback)
