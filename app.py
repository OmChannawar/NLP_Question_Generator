"""
app.py – Streamlit front-end for QuizForge (automatic question generation from textbooks).
Run with:  streamlit run app.py
"""
import html
import inspect
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from qgen.extractor import ANALYSIS_STAGES, build_chunks, extract_text, outline
from qgen.feedback import (FEEDBACK_STAGES, STAGE_LOAD as FB_STAGE_LOAD, STAGE_SUMMARY, FeedbackEntry,
                           SentimentAnalyzer, feedback_to_csv, run_sentiment, summarize_feedback)
from qgen.formatting import answer_lines, item_to_text, question_lines
from qgen.generator import ALLOWED_COUNTS, MODEL_OPTIONS, QuestionGenerator
from qgen.pipeline import STAGE_LOAD, generate_quiz, generation_stages
from qgen.progress import ProgressTracker
from qgen.schema import LLM_TYPES, QUESTION_TYPES, allocate

st.set_page_config(page_title="QuizForge – AI Question Generator", page_icon="📚", layout="wide")

# ------------------------------------------------------------------------------ styling
ASSETS = Path(__file__).parent / "assets"
OLIVE = "#556B2F"   # palette: olive green #556B2F · cream #FCF8D9 · brown #5B4733 (see .streamlit/config.toml)


def load_css(name: str) -> str:
    """Read a stylesheet from assets/ ('' if the file is missing, so the app still runs)."""
    path = ASSETS / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


ss = st.session_state
for key, default in dict(doc_key=None, chunks=None, doc_error=None, quiz_items=[], difficulty="Mid", show_answers=False,
                         quiz_id=0, ratings={}, feedback=None).items():
    ss.setdefault(key, default)


# ------------------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def load_generator(model_name: str) -> QuestionGenerator:
    """Load the Seq2Seq model once and reuse it across Streamlit reruns."""
    return QuestionGenerator(model_name)


@st.cache_resource(show_spinner=False)
def load_sentiment() -> SentimentAnalyzer:
    """Load the DistilBERT sentiment model once."""
    return SentimentAnalyzer()


class StatusPanel:
    """
    A status box that shows (1) a live progress bar with the NAME of the current process and its detail, and
    (2) a checklist of all stages: ✅ finished, ⏳ running, ▫️ waiting.
    """

    def __init__(self, title: str, stages):
        self.status = st.status(title, expanded=True)
        with self.status:
            self.bar = st.progress(0.0, text="Starting…")
            self.steps = st.empty()
        self.tracker = ProgressTracker(stages, self._render)
        self._render(self.tracker)

    def _render(self, t: ProgressTracker) -> None:
        detail = f" — {t.detail}" if t.detail else ""
        self.bar.progress(min(max(t.overall, 0.0), 1.0), text=f"{t.current}{detail}  ·  {int(t.overall * 100)} %")
        done = set(t.done)
        self.steps.markdown("  \n".join(     # two spaces + newline = a real line break in Markdown
            f"{'✅' if n in done else ('⏳' if n == t.current else '▫️')} {n}" for n, _ in t.stages))

    def finish(self, label: str) -> None:
        self.tracker.complete()
        self.status.update(label=label, state="complete", expanded=False)

    def fail(self, label: str) -> None:
        self.status.update(label=label, state="error", expanded=True)


def _save_stars(key: str, uid: int) -> None:
    """Callback: remember a star rating (widget values are lost when a question is filtered out of view)."""
    value = ss.get(key)
    if isinstance(value, int):
        value = value + 1 if hasattr(st, "feedback") else value   # st.feedback returns 0-4
    ss.ratings.setdefault(uid, {})["stars"] = value


def _save_comment(key: str, uid: int) -> None:
    ss.ratings.setdefault(uid, {})["comment"] = ss.get(key, "")


def rating_widgets(uid: int) -> None:
    """1-5 star relevance rating plus an optional comment for one question."""
    saved = ss.ratings.get(uid, {})
    c1, c2 = st.columns([1, 2])
    with c1:
        key = f"stars_{ss.quiz_id}_{uid}"
        st.caption("How relevant is this question to its topic?")
        if hasattr(st, "feedback"):
            extra = {}
            if "default" in inspect.signature(st.feedback).parameters and saved.get("stars"):
                extra["default"] = saved["stars"] - 1
            st.feedback("stars", key=key, on_change=_save_stars, args=(key, uid), **extra)
        else:  # very old Streamlit: simple radio fallback
            st.radio("Relevance (1–5)", [1, 2, 3, 4, 5], horizontal=True, key=key,
                     index=(saved["stars"] - 1) if saved.get("stars") else None,
                     on_change=_save_stars, args=(key, uid), label_visibility="collapsed")
    with c2:
        ckey = f"cmt_{ss.quiz_id}_{uid}"
        st.caption("Comment (optional)")
        st.text_input("Comment", value=saved.get("comment", ""), key=ckey, label_visibility="collapsed",
                      placeholder="e.g. clear and on topic / confusing and unrelated",
                      on_change=_save_comment, args=(ckey, uid))


def render_item(item, show_answers: bool) -> None:
    """One question card (stem, options/columns, optional answer + keywords) followed by the rating widgets."""
    lines = question_lines(item)
    rest = "".join(f'<div class="{"opt-head" if l.startswith("Column ") else "opt"}">{html.escape(l)}</div>'
                   for l in lines[1:])
    tags = (f'<span class="tag">{html.escape(QUESTION_TYPES[item.qtype])}</span>'
            f'<span class="tag topic">{html.escape(item.topic)}</span>')
    st.markdown(f'<div class="q-card">{tags}<br><span class="num">Q{item.uid + 1}.</span>'
                f'{html.escape(lines[0])}{rest}</div>', unsafe_allow_html=True)
    if show_answers:
        chips = "".join(f'<span class="chip">{html.escape(k)}</span>' for k in item.keywords) \
                or "<em>No keywords found</em>"
        st.markdown(chips, unsafe_allow_html=True)
        with st.expander("Answer & source sentence"):
            for line in answer_lines(item):
                st.write(line)
            if item.evidence and item.evidence not in (item.answer, item.question):
                st.caption("Source: " + item.evidence)
    rating_widgets(item.uid)


def group_items(items, mode: str):
    """Return [(header or None, [items])] for the chosen sorting mode."""
    if mode == "Document order":
        return [(None, items)]
    groups = {}
    for it in items:
        if mode == "Topic (A–Z)":
            label = it.topic
        elif mode == "Chapter → Topic":
            label = f"{it.chapter} › {it.topic}" if it.chapter else it.topic
        else:
            label = QUESTION_TYPES[it.qtype]
        groups.setdefault(label, []).append(it)
    if mode == "Topic (A–Z)":
        return sorted(groups.items(), key=lambda kv: kv[0].lower())
    if mode == "Question type":
        order = list(QUESTION_TYPES.values())
        return sorted(groups.items(), key=lambda kv: order.index(kv[0]))
    return list(groups.items())          # chapter → topic: order of first appearance in the book


def build_export_text(items, difficulty: str, with_answers: bool) -> str:
    """Plain-text version of the quiz for the download button."""
    head = [f"Generated questions (difficulty: {difficulty})", ""]
    return "\n".join(head + [item_to_text(it, it.uid + 1, with_answers) + "\n" for it in items])


# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Model")
    model_label = st.selectbox("Seq2Seq LLM", list(MODEL_OPTIONS), index=1)
    st.caption("The first run downloads the model weights from Hugging Face. "
               "Larger models give better questions but are slower.")
    mode = st.radio("Generation mode", ["Fast (greedy)", "Better quality (beam search, slower)"],
                    help="Beam search explores 4 candidates per word: usually nicer questions, roughly 2-4x slower.")
    num_beams = 1 if mode.startswith("Fast") else 4
    if torch.cuda.is_available():
        st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("No GPU detected – running on CPU, which is slow. Install a CUDA build of PyTorch "
                   "if you have an NVIDIA GPU, or choose Flan-T5 Small.")
    st.divider()
    soft_ui = st.toggle("Soft (neumorphic) theme", value=True,
                        help="Turn off for a flat, higher-contrast look (better for accessibility).")
    st.divider()
    st.markdown("**Pipeline**\n\n1. Read PDF/DOCX\n2. Detect chapters & topics\n3. Choose answers from the text\n"
                "4. Flan-T5 writes questions (+ MCQ/blank/match builders)\n5. Quality checks\n"
                "6. Ratings → sentiment analysis")

st.markdown(f"<style>{load_css('base.css')}{load_css('neumorphic.css') if soft_ui else ''}</style>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------------------ header
st.markdown("""<div class="hero"><h1>📚 QuizForge</h1>
<p>Upload a textbook, pick topics, question types and difficulty – get exam-style questions you can sort by topic.</p></div>""",
            unsafe_allow_html=True)

# ------------------------------------------------------------------------------ 1 · upload & analyse
st.subheader("1 · Upload your textbook")
uploaded = st.file_uploader("PDF or Word file (whole books are fine)", type=["pdf", "docx"])

if uploaded is not None:
    doc_key = (uploaded.name, uploaded.size)
    if ss.doc_key != doc_key:           # analyse each new file once
        panel = StatusPanel("Analyzing document", ANALYSIS_STAGES)
        try:
            raw = extract_text(uploaded.name, uploaded.getvalue(), panel.tracker)
            chunks = build_chunks(raw, progress=panel.tracker)
            if not chunks:
                raise ValueError("No readable text found. The file may be scanned images or too short.")
            ss.update(doc_key=doc_key, chunks=chunks, doc_error=None)
            panel.finish(f"Document ready – {len(chunks)} passages")
        except Exception as exc:
            ss.update(doc_key=doc_key, chunks=None, doc_error=str(exc))
            panel.fail("Could not read the document")
else:
    ss.update(doc_key=None, chunks=None, doc_error=None)

if ss.doc_error:
    st.error(f"Could not read the document: {ss.doc_error}")

# ------------------------------------------------------------------------------ 2 · scope
chunks = ss.chunks
scope = None
if chunks:
    book = outline(chunks)
    chapters = [c for c in book if c]
    n_topics = sum(len(t) for t in book.values())
    st.subheader("2 · Choose what to cover")
    st.caption(f"Detected {len(chapters) if chapters else 'no'} chapters · {n_topics} topics · {len(chunks)} passages")
    if not st.checkbox("Whole document", value=True):
        if chapters:
            picked = st.multiselect("Chapters", chapters)
            topic_keys = [f"{ch} › {t}" for ch in picked for t in book[ch]]
            narrowed = st.multiselect("Narrow to specific topics (optional – empty means all topics of the chapters)",
                                      topic_keys)
            scope = set(narrowed) if narrowed else set(topic_keys)
        else:
            scope = set(st.multiselect("Topics", list(book[""])))
        if not scope:
            st.warning("Select at least one chapter or topic.")

# ------------------------------------------------------------------------------ 3 · settings
st.subheader("3 · Choose question types, difficulty and number")
labels_to_types = {v: k for k, v in QUESTION_TYPES.items()}
chosen = st.multiselect("Question types", list(QUESTION_TYPES.values()), default=[QUESTION_TYPES["subjective"]])
qtypes = [labels_to_types[c] for c in chosen]
c1, c2 = st.columns(2)
with c1:
    difficulty = st.select_slider("Difficulty level", options=["Low", "Mid", "High"], value="Mid",
                                  help="Low = recall · Mid = understanding · High = analysis / harder wrong options")
with c2:
    num_q = st.select_slider("Number of questions (multiples of 5, max 30)", options=ALLOWED_COUNTS, value=10)
if qtypes:
    st.caption("Split: " + ", ".join(f"{n} × {QUESTION_TYPES[t]}" for t, n in allocate(num_q, qtypes).items()))

ready = bool(chunks) and bool(qtypes) and (scope is None or len(scope) > 0)
generate_clicked = st.button("✨ Generate questions", type="primary", disabled=not ready)

# ------------------------------------------------------------------------------ generation
if generate_clicked and ready:
    panel = StatusPanel("Generating questions", generation_stages(qtypes))
    try:
        generator = None
        if any(t in LLM_TYPES for t in qtypes):
            panel.tracker.update(STAGE_LOAD, 0.05, "first run downloads the model weights")
            generator = load_generator(MODEL_OPTIONS[model_label])
            panel.tracker.finish(STAGE_LOAD, f"running on {generator.device.upper()}")
        items = generate_quiz(chunks, generator, difficulty.lower(), num_q, qtypes, scope,
                              progress=panel.tracker, num_beams=num_beams)
        ss.update(quiz_items=items, show_answers=False, difficulty=difficulty, ratings={}, feedback=None,
                  quiz_id=ss.quiz_id + 1)
        panel.finish(f"Done – {len(items)} questions")
        want, got = allocate(num_q, qtypes), {t: sum(i.qtype == t for i in items) for t in qtypes}
        short = [f"{QUESTION_TYPES[t]}: {got[t]} of {want[t]}" for t in qtypes if got[t] < want[t]]
        if short:
            st.warning("Some types produced fewer questions than requested because the text had too few suitable "
                       "sentences/definitions – " + "; ".join(short))
    except Exception as exc:  # friendly message instead of a stack trace
        panel.fail("Generation failed")
        st.error(f"Could not generate questions: {exc}")

# ------------------------------------------------------------------------------ 4 · results
items = ss.quiz_items   # (not ss.items: that is the dict method)
if items:
    st.subheader(f"4 · Your questions ({ss.difficulty} difficulty)")

    f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
    with f1:
        sort_mode = st.selectbox("Sort / group by", ["Document order", "Topic (A–Z)", "Chapter → Topic",
                                                     "Question type"])
    with f2:
        topic_filter = st.multiselect("Show only these topics (optional)", sorted({i.topic for i in items}))
    with f3:
        type_filter = st.multiselect("Show only these types (optional)",
                                     [QUESTION_TYPES[t] for t in QUESTION_TYPES if any(i.qtype == t for i in items)])
    shown = [i for i in items
             if (not topic_filter or i.topic in topic_filter)
             and (not type_filter or QUESTION_TYPES[i.qtype] in type_filter)]
    st.caption(f"Showing {len(shown)} of {len(items)} questions")

    if not ss.show_answers:
        if st.button("🔑 Answer the questions (show keywords)"):
            ss.show_answers = True
            st.rerun()
    elif st.button("🙈 Hide keywords"):
        ss.show_answers = False
        st.rerun()

    for header, group in group_items(shown, sort_mode):
        if header is not None:
            st.markdown(f"#### 📂 {header}  ·  {len(group)}")
        for it in group:
            render_item(it, ss.show_answers)

    st.download_button("⬇️ Download questions (.txt)", build_export_text(items, ss.difficulty, ss.show_answers),
                       file_name="questions.txt")

    # -------------------------------------------------------------------------- 5 · feedback + sentiment
    st.subheader("5 · Rate the questions & analyze feedback")
    st.caption("Give each question 1–5 stars for how relevant it is to its topic (1 = unrelated, 5 = perfectly on topic) "
               "and, if you like, a short comment. The comments are classified by a Transformer sentiment model.")
    if st.button("📊 Analyze feedback (sentiment analysis)"):
        entries = [FeedbackEntry(it.uid, it.question, it.qtype, it.topic,
                                 ss.ratings.get(it.uid, {}).get("stars"),
                                 (ss.ratings.get(it.uid, {}).get("comment") or "").strip()) for it in items]
        if not any(e.stars or e.comment for e in entries):
            st.warning("Rate at least one question (stars or comment) first.")
        else:
            panel = StatusPanel("Analyzing feedback", FEEDBACK_STAGES)
            try:
                analyzer = None
                if any(e.comment for e in entries):
                    panel.tracker.update(FB_STAGE_LOAD, 0.05, "first run downloads ~270 MB")
                    analyzer = load_sentiment()
                panel.tracker.finish(FB_STAGE_LOAD)
                run_sentiment(entries, analyzer, panel.tracker)
                panel.tracker.update(STAGE_SUMMARY, 0.5, "aggregating ratings")
                ss.feedback = (entries, summarize_feedback(entries))
                panel.finish("Feedback analysed")
            except Exception as exc:
                panel.fail("Feedback analysis failed")
                st.error(f"Could not analyze feedback: {exc}")

    if ss.feedback:
        entries, summary = ss.feedback
        m1, m2, m3 = st.columns(3)
        m1.metric("Average relevance", f"{summary.avg_stars:.1f} / 5" if summary.avg_stars else "–")
        m2.metric("Questions rated", summary.n_rated)
        m3.metric("Comments analysed", summary.n_commented)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Star ratings**")
            st.bar_chart(pd.DataFrame({"questions": [summary.star_counts.get(s, 0) for s in range(1, 6)]},
                                      index=[f"{s}★" for s in range(1, 6)]), color=OLIVE)
        with g2:
            st.markdown("**Comment sentiment**")
            st.bar_chart(pd.DataFrame({"comments": [summary.sentiment_counts.get(k, 0)
                                                    for k in ("positive", "mixed", "negative")]},
                                      index=["positive", "mixed", "negative"]), color=OLIVE)
        if summary.by_topic:
            st.markdown("**Average relevance by topic (weakest first)**")
            st.dataframe(pd.DataFrame(summary.by_topic, columns=["Topic", "Avg ★", "Rated"]).round({"Avg ★": 2}),
                         hide_index=True)
        if summary.flagged:
            st.markdown("**⚠️ Questions to review**")
            by_uid = {e.uid: e for e in entries}
            for uid, reason in summary.flagged:
                st.write(f"Q{uid + 1} – {reason}: {by_uid[uid].question[:140]}")
        st.download_button("⬇️ Download feedback (.csv)", feedback_to_csv(entries), file_name="feedback.csv")
