"""
feedback.py
-----------
Question-quality feedback with SENTIMENT ANALYSIS.

Each question can be rated 1-5 stars for how relevant it is to its topic and may carry a short comment.
The comments are classified by a Transformer sentiment model (DistilBERT fine-tuned on SST-2), and the
ratings + sentiments are summarised overall, per topic and per question type; weak questions are flagged.
"""
import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .progress import ProgressTracker, tick

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
STAGE_LOAD = "Loading sentiment model"
STAGE_ANALYZE = "Analyzing comments"
STAGE_SUMMARY = "Summarizing feedback"
FEEDBACK_STAGES = [(STAGE_LOAD, 1.5), (STAGE_ANALYZE, 2.0), (STAGE_SUMMARY, 0.3)]
CONFIDENCE_THRESHOLD = 0.6   # below this the comment is reported as "mixed"


@dataclass
class FeedbackEntry:
    """Rating + comment for one question."""
    uid: int
    question: str
    qtype: str
    topic: str
    stars: Optional[int] = None          # 1-5, None = not rated
    comment: str = ""
    sentiment: Optional[str] = None      # 'positive' | 'negative' | 'mixed' | None (no comment)
    confidence: float = 0.0


@dataclass
class FeedbackSummary:
    """Aggregated feedback."""
    n_rated: int = 0
    n_commented: int = 0
    avg_stars: Optional[float] = None
    star_counts: Dict[int, int] = field(default_factory=dict)
    sentiment_counts: Dict[str, int] = field(default_factory=dict)
    by_topic: List[Tuple[str, float, int]] = field(default_factory=list)   # (topic, avg stars, n rated)
    by_type: List[Tuple[str, float, int]] = field(default_factory=list)
    flagged: List[Tuple[int, str]] = field(default_factory=list)           # (uid, reason)


def classify(label: str, score: float) -> str:
    """Map a model output to positive / negative / mixed using a confidence threshold."""
    if score < CONFIDENCE_THRESHOLD:
        return "mixed"
    return "positive" if label.upper().startswith("POS") else "negative"


class SentimentAnalyzer:
    """Wraps the Hugging Face sentiment-analysis pipeline (DistilBERT SST-2)."""

    def __init__(self, model_name: str = SENTIMENT_MODEL):
        """Load the model (downloads ~270 MB on first use); runs on the GPU when available."""
        import torch
        from transformers import pipeline
        self.pipe = pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def analyze(self, texts: List[str], progress: Optional[ProgressTracker] = None,
                batch_size: int = 8) -> List[Tuple[str, float]]:
        """Return (sentiment, confidence) for every text; progress is reported per batch."""
        results: List[Tuple[str, float]] = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            for out in self.pipe(batch, truncation=True, max_length=128):
                results.append((classify(out["label"], float(out["score"])), float(out["score"])))
            tick(progress, STAGE_ANALYZE, min(s + batch_size, len(texts)) / max(len(texts), 1),
                 f"comment {min(s + batch_size, len(texts))} of {len(texts)}")
        return results


def run_sentiment(entries: List[FeedbackEntry], analyzer, progress: Optional[ProgressTracker] = None) -> None:
    """Classify the comment of every entry that has one (in place)."""
    commented = [e for e in entries if e.comment.strip()]
    tick(progress, STAGE_ANALYZE, 0.0, f"0 of {len(commented)} comments")
    if commented:
        for e, (label, conf) in zip(commented, analyzer.analyze([e.comment.strip() for e in commented], progress)):
            e.sentiment, e.confidence = label, conf
    tick(progress, STAGE_ANALYZE, 1.0, f"{len(commented)} comments analysed")


def _group_average(entries: List[FeedbackEntry], key) -> List[Tuple[str, float, int]]:
    groups: Dict[str, List[int]] = {}
    for e in entries:
        if e.stars:
            groups.setdefault(key(e), []).append(e.stars)
    return sorted(((g, sum(v) / len(v), len(v)) for g, v in groups.items()), key=lambda x: x[1])


def summarize_feedback(entries: List[FeedbackEntry]) -> FeedbackSummary:
    """
    Aggregate ratings and sentiments. A question is flagged when it got 1-2 stars or a negative comment;
    the lists are sorted from weakest to strongest so the worst topics are shown first.
    """
    rated = [e for e in entries if e.stars]
    summary = FeedbackSummary(
        n_rated=len(rated),
        n_commented=sum(1 for e in entries if e.sentiment),
        avg_stars=(sum(e.stars for e in rated) / len(rated)) if rated else None,
        star_counts={s: sum(1 for e in rated if e.stars == s) for s in range(1, 6)},
        sentiment_counts={k: sum(1 for e in entries if e.sentiment == k) for k in ("positive", "mixed", "negative")},
        by_topic=_group_average(entries, lambda e: e.topic),
        by_type=_group_average(entries, lambda e: e.qtype),
    )
    for e in entries:
        reasons = []
        if e.stars and e.stars <= 2:
            reasons.append(f"low relevance rating ({e.stars}★)")
        if e.sentiment == "negative":
            reasons.append("negative comment")
        if reasons:
            summary.flagged.append((e.uid, " + ".join(reasons)))
    return summary


def feedback_to_csv(entries: List[FeedbackEntry]) -> str:
    """CSV text of all feedback (for the download button)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["question_no", "type", "topic", "question", "stars", "comment", "sentiment", "confidence"])
    for e in entries:
        writer.writerow([e.uid + 1, e.qtype, e.topic, e.question, e.stars or "", e.comment,
                         e.sentiment or "", f"{e.confidence:.2f}" if e.sentiment else ""])
    return buf.getvalue()
