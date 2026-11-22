"""
generator.py
------------
Neural question-generation core (answer-aware).

NLP concepts used
  * Seq2Seq      : Flan-T5 is an encoder-decoder (sequence-to-sequence) model. The encoder reads the
                   instruction + answer + textbook passage; the decoder writes the question token by token.
  * Transformers : the architecture (self-/cross-attention) and the Hugging Face `transformers` library.
  * LLM          : Flan-T5 is an instruction-tuned language model, so the *difficulty level* is controlled by
                   the natural-language prompt (prompt engineering).

Quality safeguards (why questions are no longer "random")
  1. Answer-aware generation: the answer is taken from the textbook first (see candidates.py), then the LLM
     only has to write a question for it -> the question is answerable by construction.
  2. Grounding check: a question whose key words are not found in the passage is rejected (hallucination).
  3. Leak check: a question that already contains its own short answer is rejected.
  4. Duplicate check on content words; rejected slots are retried with the next candidate sentence.
"""
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .candidates import Candidate
from .keywords import extract_keywords
from .schema import ALLOWED_COUNTS, DIFFICULTY_LEVELS, QAItem, validate_request  # noqa: F401 (re-exported)

# --------------------------------------------------------------------------- configuration
MODEL_OPTIONS = {
    "Flan-T5 Small  (fastest, ~80M params)": "google/flan-t5-small",
    "Flan-T5 Base   (recommended, ~250M)": "google/flan-t5-base",
    "Flan-T5 Large  (best quality, ~780M)": "google/flan-t5-large",
}
MAX_ATTEMPTS = 4  # rounds per question slot; later rounds use the next candidate / sampling

# Style hint per question type and difficulty level (aligned with Bloom's taxonomy).
STYLE_HINTS = {
    "subjective": {
        "low": "The question must be a simple factual question starting with What, Which, Who or Where.",
        "mid": "The question must ask the student to explain or describe an idea, for example starting with "
               "How or Explain.",
        "high": "The question must be challenging and need reasoning, such as why something works or how "
                "two things compare.",
    },
    "mcq": {  # the answer is a single term; the question must not name it
        "low": "The question must be a simple factual question starting with What or Which that has one short answer.",
        "mid": "The question must describe the purpose or role of the concept and ask which concept it is.",
        "high": "The question must describe a situation or property that needs reasoning to identify the concept.",
    },
}
# words that appear in questions but carry no topic information
QUESTION_FILLER = {
    # instruction verbs
    "explain", "describe", "compare", "evaluate", "define", "discuss", "justify", "analyse", "analyze",
    "identify", "state", "list", "give", "contrast", "differentiate", "illustrate", "summarize",
    # generic question vocabulary that is rarely copied from a passage
    "main", "following", "term", "mean", "means", "meant", "called", "name", "role", "purpose", "reason",
    "reasons", "work", "works", "working", "practice", "important", "importance", "significant",
    "significance", "benefit", "benefits", "affect", "affects", "impact", "useful", "matter", "matters",
    "help", "helps", "ensure", "differ", "difference", "differences", "similar", "similarity",
    "relationship", "relate", "characteristic", "characteristics", "key", "idea", "concept", "process",
    "step", "steps", "example", "examples", "typically", "generally", "different", "type", "types",
}
# a question passes if at least this share of its topic words occur in the passage
MIN_GROUNDING = 0.6


# --------------------------------------------------------------------------- validation helpers
def _tokens(text: str) -> set:
    """Lower-cased content-word set (filler words removed) used for duplicate detection."""
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


def _stem(word: str) -> str:
    """Very light plural stemmer ('pheromones' -> 'pheromone', 'ies' -> 'y')."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _content_stems(text: str) -> set:
    """Stemmed topic words of a text (stop-words and question filler removed)."""
    words = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return {_stem(w) for w in words if w not in ENGLISH_STOP_WORDS and w not in QUESTION_FILLER}


def is_grounded(question: str, cand: Candidate, min_overlap: float = MIN_GROUNDING) -> bool:
    """
    Accept a question only if
      * at least `min_overlap` of its topic words occur in the passage (otherwise the model made something up), and
      * it does not already contain its own (short) answer.
    """
    q = _content_stems(question)
    if not q:
        return False
    context = _content_stems(cand.context + " " + cand.title)
    if len(q & context) / len(q) < min_overlap:
        return False
    a = _content_stems(cand.answer)
    if a and len(cand.answer.split()) <= 4 and a <= q:
        return False
    return True


def make_keywords(cand: Candidate) -> List[str]:
    """Keywords of the answer: the term itself (term answers) plus TF-IDF key words of the supporting sentence."""
    if cand.kind != "term":
        return extract_keywords(cand.answer, cand.context, top_k=6)
    term = cand.answer.lower()
    return [term] + [k for k in extract_keywords(cand.evidence, cand.context, top_k=5) if k not in term][:5]


# --------------------------------------------------------------------------- model wrapper
class QuestionGenerator:
    """Wraps a Seq2Seq Transformer LLM and exposes batched, answer-aware question generation."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Download (first run only) and load the tokenizer and model.
        On an NVIDIA GPU the model runs in bfloat16 (faster, half the memory); on CPU in float32.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    @staticmethod
    def build_question_prompt(cand: Candidate, difficulty: str, qtype: str = "subjective") -> str:
        """
        Answer-aware instruction prompt: tells the LLM which answer the question must have, the wanted
        style (difficulty) and gives the passage (with its section title) to read.
        """
        passage = f"{cand.title}. {cand.context}" if cand.title else cand.context
        return (f"{STYLE_HINTS[qtype][difficulty]} Generate a question that has the following answer: "
                f"{cand.answer} for the following passage: {passage}")

    def _generate_batch(self, prompts: List[str], max_new_tokens: int, sample: bool = False,
                        num_beams: int = 1) -> List[str]:
        """
        Encode several prompts at once (padded), run the encoder-decoder once for the whole batch and decode.
        sample=False -> greedy (num_beams=1) or beam search (num_beams>1); sample=True -> nucleus sampling.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        gen_args = (dict(do_sample=True, top_p=0.92, temperature=0.9) if sample
                    else dict(do_sample=False, num_beams=num_beams))
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         no_repeat_ngram_size=3, **gen_args)
        return [t.strip() for t in self.tokenizer.batch_decode(output, skip_special_tokens=True)]

    @staticmethod
    def _clean_question(raw: str) -> str:
        """Tidy model output into a proper question; returns '' if unusable (too short)."""
        q = raw.strip().strip('"')
        if len(q.split()) < 4:
            return ""
        q = q[0].upper() + q[1:]
        return q if q.endswith("?") else q.rstrip(".") + "?"

    def generate_question(self, cand: Candidate, difficulty: str, sample: bool = False, num_beams: int = 1,
                          qtype: str = "subjective") -> str:
        """Generate one question for one candidate (handy for demos / the notebook; no validation)."""
        raw = self._generate_batch([self.build_question_prompt(cand, difficulty, qtype)], 48, sample, num_beams)[0]
        return self._clean_question(raw)

    def generate_items(self, slots: List[List[Candidate]], difficulty: str, num_beams: int = 1,
                       batch_size: Optional[int] = None, qtype: str = "subjective",
                       progress_callback: Optional[Callable[[float], None]] = None) -> List[QAItem]:
        """
        Turn planned slots into validated questions.

        Each round, every unfilled slot tries its next candidate (round 1: best sentence; later rounds:
        alternative sentences from the same passage, or sampling if there is none).  A question is accepted
        only if it passes the grounding, leak and duplicate checks.  Questions are generated in batches.
        Returns items in document order; slots that never produce a valid question are skipped.
        """
        batch_size = batch_size or (8 if self.device == "cuda" else 4)
        n = len(slots)
        results: List[Optional[QAItem]] = [None] * n

        for attempt in range(MAX_ATTEMPTS):
            pending = [i for i, r in enumerate(results) if r is None]
            if not pending:
                break
            # (slot index, candidate, use_sampling): sample only when no new candidate is left to try
            jobs = [(i, slots[i][min(attempt, len(slots[i]) - 1)], attempt >= len(slots[i])) for i in pending]
            for use_sampling in (False, True):
                group = [j for j in jobs if j[2] == use_sampling]
                for s in range(0, len(group), batch_size):
                    batch = group[s:s + batch_size]
                    prompts = [self.build_question_prompt(c, difficulty, qtype) for _, c, _ in batch]
                    outputs = self._generate_batch(prompts, 48, sample=use_sampling, num_beams=num_beams)
                    for (i, cand, _), raw in zip(batch, outputs):
                        q = self._clean_question(raw)
                        accepted = [r.question for r in results if r]
                        if q and is_grounded(q, cand) and not is_duplicate(q, accepted):
                            results[i] = QAItem(q, cand.answer, make_keywords(cand), cand.evidence, qtype,
                                            cand.chapter, cand.title, cand.chunk_id)
                    if progress_callback:
                        progress_callback(sum(r is not None for r in results) / n)
        return [r for r in results if r]
