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
from typing import Callable, List, Optional

import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
MAX_ATTEMPTS = 4  # rounds of generation; round 1 is greedy/beam, later rounds sample to avoid duplicates

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
    """
    Lower-cased *content-word* set used for duplicate detection. Filler words ("what", "is", "the", ...)
    are removed, otherwise "What is the function of X?" and "What is the function of Y?" would look identical.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    content = {w for w in words if w not in ENGLISH_STOP_WORDS}
    return content or set(words)


def is_duplicate(question: str, seen: List[str], threshold: float = 0.7) -> bool:
    """True if `question` has Jaccard content-word overlap >= threshold with any already accepted question."""
    a = _tokens(question)
    for other in seen:
        b = _tokens(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


class QuestionGenerator:
    """Wraps a Seq2Seq Transformer LLM and exposes (batched) question + answer generation."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Download (first run only) and load the tokenizer and model.
        On an NVIDIA GPU the model runs in bfloat16 (about 2x faster, half the memory);
        on CPU it runs in float32.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

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
    def _generate_batch(self, prompts: List[str], max_new_tokens: int, sample: bool = False,
                        num_beams: int = 1) -> List[str]:
        """
        Encode several prompts at once (padded into one tensor), run the encoder-decoder once
        for the whole batch and decode every output. Batching is the main speed-up over
        generating one passage at a time.

        sample=False -> greedy search (num_beams=1) or beam search (num_beams>1, slower, better)
        sample=True  -> nucleus sampling (used on retries to get a *different* question)
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        if sample:
            gen_args = dict(do_sample=True, top_p=0.92, temperature=0.9)
        else:
            gen_args = dict(do_sample=False, num_beams=num_beams)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         no_repeat_ngram_size=3, **gen_args)
        return [t.strip() for t in self.tokenizer.batch_decode(output, skip_special_tokens=True)]

    @staticmethod
    def _clean_question(raw: str) -> str:
        """Tidy model output into a proper question; returns '' if it is unusable (too short)."""
        q = raw.strip().strip('"')
        if len(q.split()) < 4:
            return ""
        q = q[0].upper() + q[1:]
        return q if q.endswith("?") else q.rstrip(".") + "?"

    # ------------------------------------------------------------------ public API
    def generate_question(self, context: str, difficulty: str, sample: bool = False,
                          num_beams: int = 1) -> str:
        """Generate a single question for one passage (handy for demos / the notebook)."""
        raw = self._generate_batch([self.build_question_prompt(context, difficulty)], 48, sample, num_beams)[0]
        return self._clean_question(raw)

    def generate_items(self, contexts: List[str], difficulty: str, num_beams: int = 1,
                       batch_size: Optional[int] = None,
                       progress_callback: Optional[Callable[[float], None]] = None) -> List[QAItem]:
        """
        Generate one unique question + answer + keywords for every passage in `contexts`.

        Stage 1 (questions): passages are processed in batches. Passages whose question was empty or a
                             near-duplicate are retried in later rounds using sampling.
        Stage 2 (answers):   answers for all accepted questions are generated in batches, then keywords
                             are extracted with TF-IDF.

        progress_callback receives a float in [0, 1]. Passages that never yield a valid question are skipped,
        so fewer items than passages may be returned.
        """
        batch_size = batch_size or (8 if self.device == "cuda" else 4)
        n = len(contexts)
        questions: List[Optional[str]] = [None] * n

        def report(fraction: float) -> None:
            if progress_callback:
                progress_callback(min(fraction, 1.0))

        # ---- stage 1: questions
        for attempt in range(MAX_ATTEMPTS):
            pending = [i for i, q in enumerate(questions) if q is None]
            if not pending:
                break
            for s in range(0, len(pending), batch_size):
                idxs = pending[s:s + batch_size]
                prompts = [self.build_question_prompt(contexts[i], difficulty) for i in idxs]
                outputs = self._generate_batch(prompts, 48, sample=attempt > 0, num_beams=num_beams)
                for i, raw in zip(idxs, outputs):
                    q = self._clean_question(raw)
                    if q and not is_duplicate(q, [x for x in questions if x]):
                        questions[i] = q
                report(0.6 * sum(q is not None for q in questions) / n)

        # ---- stage 2: answers + keywords
        keep = [i for i, q in enumerate(questions) if q]
        answers = {}
        for s in range(0, len(keep), batch_size):
            idxs = keep[s:s + batch_size]
            prompts = [self.build_answer_prompt(contexts[i], questions[i]) for i in idxs]
            for i, ans in zip(idxs, self._generate_batch(prompts, 64)):
                answers[i] = ans
            report(0.6 + 0.4 * min(s + batch_size, len(keep)) / len(keep))

        return [QAItem(questions[i], answers[i], extract_keywords(answers[i], contexts[i])) for i in keep]
