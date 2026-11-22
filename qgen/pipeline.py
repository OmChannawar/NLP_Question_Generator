"""
pipeline.py
-----------
End-to-end orchestration with named, weighted progress stages:

    text -> chapters/topics -> choose scope -> plan answers -> build each question type -> sort -> QAItems

Question types
  subjective, mcq   : question text written by the Seq2Seq LLM (answer-aware, quality-checked)
  fill_blank, true_false, match : built from the textbook sentences with NLP rules (no hallucination possible)
"""
import random
from typing import Callable, Iterable, List, Optional, Sequence, Union

from .candidates import TermEntry, TermPool, build_idf, chunk_candidates, plan_slots
from .extractor import Chunk, build_chunks
from .generator import QuestionGenerator
from .progress import ProgressTracker, tick
from .qtypes import add_mcq_options, build_fill_blank, build_match, build_true_false
from .schema import LLM_TYPES, QUESTION_TYPES, QAItem, allocate, validate_request

STAGE_LOAD = "Loading AI model"
STAGE_PLAN = "Planning questions"
STAGE_FINAL = "Finalizing & sorting"


def stage_for(qtype: str) -> str:
    """Name of the progress stage that builds one question type."""
    return f"Generating {QUESTION_TYPES[qtype]}"


def generation_stages(qtypes: Iterable[str], include_model_load: Optional[bool] = None):
    """The ordered (name, weight) stages shown in the UI for the chosen question types."""
    types = [t for t in QUESTION_TYPES if t in set(qtypes)]
    if include_model_load is None:
        include_model_load = any(t in LLM_TYPES for t in types)
    stages = [(STAGE_LOAD, 1.5)] if include_model_load else []
    stages.append((STAGE_PLAN, 1.0))
    stages += [(stage_for(t), 4.0 if t in LLM_TYPES else 1.0) for t in types]
    stages.append((STAGE_FINAL, 0.3))
    return stages


def generate_quiz(source: Union[str, Sequence[Chunk]], generator: Optional[QuestionGenerator], difficulty: str,
                  num_questions: int, qtypes: Sequence[str] = ("subjective",), scope: Optional[Iterable[str]] = None,
                  progress: Optional[ProgressTracker] = None, num_beams: int = 1, seed: int = 7,
                  progress_callback: Optional[Callable[[float], None]] = None) -> List[QAItem]:
    """
    Generate `num_questions` questions split over the chosen `qtypes`.

    source   : raw text from extract_text() or the list of Chunks from build_chunks()
    scope    : optional set of Chunk.topic_key values; only those topics are asked about
               (distractors / wrong options may still come from anywhere in the book)
    progress : ProgressTracker shown by the UI;  progress_callback(float) is a simpler alternative
    num_beams: 1 = fast greedy decoding, 4 = beam search (better wording, slower)
    May return fewer questions than requested (e.g. too few definitions for Match the pair).
    """
    validate_request(num_questions, difficulty)
    types = [t for t in QUESTION_TYPES if t in set(qtypes)]
    if not types:
        raise ValueError("Choose at least one question type.")
    if any(t in LLM_TYPES for t in types) and generator is None:
        raise ValueError("A language model is required for Subjective and MCQ questions.")
    if progress is None and progress_callback is not None:
        progress = ProgressTracker(generation_stages(types, include_model_load=False),
                                   lambda tr: progress_callback(tr.overall))

    chunks = build_chunks(source) if isinstance(source, str) else list(source)
    if not chunks:
        raise ValueError("No readable prose found. The file may be scanned images or too short.")
    wanted = set(scope) if scope is not None else None
    scope_chunks = [c for c in chunks if wanted is None or c.topic_key in wanted]
    if not scope_chunks:
        raise ValueError("The selected chapters/topics contain no readable text.")
    rng = random.Random(seed)

    # ---- planning: score sentences once, build the term pool used for wrong options / pairs
    tick(progress, STAGE_PLAN, 0.02, "scoring sentences")
    idf = build_idf(chunks)
    cache: dict = {}
    terms_by_chunk = cache.setdefault(("term", "-"), {})
    for k, c in enumerate(chunks):
        terms_by_chunk[c.id] = chunk_candidates(c, "low", idf, "term")
        if k % 25 == 0:
            tick(progress, STAGE_PLAN, 0.05 + 0.85 * k / len(chunks), f"passage {k + 1} of {len(chunks)}")
    all_terms = [x for v in terms_by_chunk.values() for x in v]
    pool = TermPool([TermEntry(x.answer, x.evidence, x.chunk_id, x.chapter, x.title) for x in all_terms])
    scope_ids = {c.id for c in scope_chunks}
    definitions = [TermEntry(x.answer, x.evidence, x.chunk_id, x.chapter, x.title, x.defn)
                   for x in all_terms if x.defn and x.chunk_id in scope_ids]
    tick(progress, STAGE_PLAN, 1.0, "done")

    # ---- build every question type
    counts = allocate(num_questions, types)
    used: set = set()   # sentences already asked about (avoids the same fact in two question types)
    items: List[QAItem] = []
    for t in types:
        n_t, stage = counts[t], stage_for(t)
        tick(progress, stage, 0.0, f"0 of {n_t} questions")

        def report(f, s=stage, n=n_t):
            tick(progress, s, f, f"{int(round(f * n))} of {n} questions")

        if t == "subjective":
            slots = plan_slots(scope_chunks, n_t, difficulty, None, used, idf, cache)
            made = generator.generate_items(slots, difficulty, num_beams=num_beams, qtype=t, progress_callback=report)
        elif t == "mcq":
            slots = plan_slots(scope_chunks, n_t, difficulty, "term", used, idf, cache)
            made = generator.generate_items(slots, difficulty, num_beams=num_beams, qtype=t, progress_callback=report)
            made = add_mcq_options(made, pool, difficulty, rng)
        elif t == "fill_blank":
            made = build_fill_blank(plan_slots(scope_chunks, n_t, difficulty, "term", used, idf, cache), difficulty, idf)
        elif t == "true_false":
            made = build_true_false(plan_slots(scope_chunks, n_t, difficulty, "term", used, idf, cache),
                                    difficulty, pool, rng)
        else:  # match
            made = build_match([d for d in definitions if d.evidence not in used], n_t, difficulty, rng)
        used.update(m.evidence for m in made)
        items.extend(made)
        tick(progress, stage, 1.0, f"{len(made)} of {n_t} questions")

    # ---- document order (book position, then type), numbering
    tick(progress, STAGE_FINAL, 0.2, "sorting by position in the book")
    type_order = {t: i for i, t in enumerate(QUESTION_TYPES)}
    items.sort(key=lambda it: (it.chunk_id, type_order[it.qtype]))
    for i, it in enumerate(items):
        it.uid, it.topic = i, it.topic or "General"
    tick(progress, STAGE_FINAL, 1.0, f"{len(items)} questions ready")
    return items
