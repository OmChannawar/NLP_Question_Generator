"""
generator.py
------------
Neural question-generation core.

NLP concepts used
  * Seq2Seq      : Flan-T5 is an encoder-decoder (sequence-to-sequence) model. The encoder reads
                   the textbook passage + instruction; the decoder writes the question token by token.
  * Transformers : the architecture (self-attention) and the Hugging Face `transformers` library.
  * LLM          : Flan-T5 is an instruction-tuned language model, so the *difficulty level* is
                   controlled simply by changing the natural-language prompt (prompt engineering).
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .keywords import extract_keywords

# --------------------------------------------------------------------------- configuration
MODEL_OPTIONS = {
    "Flan-T5 Small  (fastest, ~80M params)": "google/flan-t5-small",
    "Flan-T5 Base   (recommended, ~250M)": "google/flan-t5-base",
    "Flan-T5 Large  (best quality, ~780M)": "google/flan-t5-large",
}
DIFFICULTY_LEVELS = ["low", "mid", "high"]
ALLOWED_COUNTS = [5, 10, 15, 20, 25, 30]  # multiples of 5, max 30

# One instruction per difficulty level (aligned with Bloom's taxonomy).
DIFFICULTY_INSTRUCTIONS = {
    "low": "Write one simple factual recall question about the passage. "
           "It must start with What, Who, When, Where or Which and be answerable by a single fact from the passage.",
    "mid": "Write one conceptual question about the passage that tests understanding. "
           "It should ask the student to explain, describe or define an idea, for example starting with How or Explain.",
    "high": "Write one challenging analytical question about the passage that needs reasoning. "
            "It should ask the student to analyse, compare, justify or evaluate, for example starting with Why or How does.",
}


@dataclass
class QAItem:
    """One generated quiz entry."""
    question: str
    answer: str
    keywords: List[str] = field(default_factory=list)


def validate_request(num_questions: int, difficulty: str) -> None:
    """Enforce the project rules: difficulty in {low, mid, high}; count is a multiple of 5, max 30."""
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"Difficulty must be one of {DIFFICULTY_LEVELS}.")
    if num_questions % 5 != 0 or not 5 <= num_questions <= 30:
        raise ValueError("Number of questions must be a multiple of 5 between 5 and 30.")


def _tokens(text: str) -> set:
    """Lower-cased word set used for duplicate detection."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_duplicate(question: str, seen: List[str], threshold: float = 0.7) -> bool:
    """True if `question` has Jaccard word-overlap >= threshold with any already accepted question."""
    a = _tokens(question)
    for other in seen:
        b = _tokens(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


class QuestionGenerator:
    """Wraps a Seq2Seq Transformer LLM and exposes question + answer generation."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Download (first run only) and load the tokenizer and model on GPU if available."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    # ------------------------------------------------------------------ prompt builders
    @staticmethod
    def build_question_prompt(context: str, difficulty: str) -> str:
        """Compose the instruction prompt that steers the LLM towards the requested difficulty."""
        return f"{DIFFICULTY_INSTRUCTIONS[difficulty]}\n\nPassage: {context}\n\nQuestion:"

    @staticmethod
    def build_answer_prompt(context: str, question: str) -> str:
        """Prompt that asks the LLM to answer using only the passage (used to derive keywords)."""
        return ("Answer the question using only the passage. "
                "Reply with a complete answer in one or two sentences.\n\n"
                f"Passage: {context}\n\nQuestion: {question}\n\nAnswer:")

    # ------------------------------------------------------------------ low-level decoding
    def _generate(self, prompt: str, max_new_tokens: int, sample: bool = False) -> str:
        """
        Encode `prompt`, run the encoder-decoder and decode the output text.
        sample=False -> beam search (stable, best quality)
        sample=True  -> nucleus sampling (used to get a *different* question on retries)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        gen_args = (dict(do_sample=True, top_p=0.92, temperature=0.9)
                    if sample else dict(num_beams=4, early_stopping=True))
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         no_repeat_ngram_size=3, **gen_args)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ public API
    def generate_question(self, context: str, difficulty: str, sample: bool = False) -> str:
        """Generate one question for `context` at the given difficulty; returns '' if output is unusable."""
        q = self._generate(self.build_question_prompt(context, difficulty), 48, sample)
        q = q.strip().strip('"')
        if len(q.split()) < 4:  # too short to be a real question
            return ""
        q = q[0].upper() + q[1:]
        return q if q.endswith("?") else q.rstrip(".") + "?"

    def generate_answer(self, context: str, question: str) -> str:
        """Let the LLM answer its own question from the passage (grounding for the keywords)."""
        return self._generate(self.build_answer_prompt(context, question), 80)

    def generate_item(self, context: str, difficulty: str, seen: Optional[List[str]] = None,
                      max_attempts: int = 4) -> Optional[QAItem]:
        """
        Produce a unique question + answer + keywords for one passage.
        The first attempt uses beam search; retries use sampling to avoid repeating a question.
        Returns None if no acceptable, non-duplicate question was produced.
        """
        seen = seen or []
        for attempt in range(max_attempts):
            question = self.generate_question(context, difficulty, sample=attempt > 0)
            if question and not is_duplicate(question, seen):
                answer = self.generate_answer(context, question)
                return QAItem(question, answer, extract_keywords(answer, context))
        return None
