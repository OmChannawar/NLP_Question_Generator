"""
pipeline.py
-----------
End-to-end orchestration:
    raw text -> section-aware passages -> answer candidates -> Seq2Seq LLM questions -> checks -> keywords
"""
from typing import Callable, List, Optional

from .candidates import plan_slots
from .extractor import build_chunks
from .generator import QAItem, QuestionGenerator, validate_request


def generate_quiz(raw_text: str, generator: QuestionGenerator, difficulty: str, num_questions: int,
                  progress_callback: Optional[Callable[[float], None]] = None,
                  num_beams: int = 1) -> List[QAItem]:
    """
    Generate up to `num_questions` questions spread across the document.

    raw_text      : text exactly as returned by extract_text() (line breaks are needed to detect headings)
    num_beams     : 1 = fast greedy decoding (default); 4 = beam search (better wording, slower)
    May return fewer questions than requested if the document has too few suitable sentences.
    """
    validate_request(num_questions, difficulty)
    chunks = build_chunks(raw_text)
    if not chunks:
        raise ValueError("No readable prose found. The file may be scanned images or too short.")
    slots = plan_slots(chunks, num_questions, difficulty)
    if not slots:
        raise ValueError("No suitable sentences found to ask questions about.")
    return generator.generate_items(slots, difficulty, num_beams=num_beams, progress_callback=progress_callback)
