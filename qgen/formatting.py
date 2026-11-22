"""
formatting.py
-------------
Turns a QAItem of any type into plain-text lines (used by the UI cards, the .txt export and the notebook).
"""
from typing import List

from .schema import QUESTION_TYPES, QAItem

LETTERS = "ABCDEFGH"


def question_lines(item: QAItem) -> List[str]:
    """Question text plus, depending on the type, its options / columns."""
    if item.qtype == "mcq":
        return [item.question] + [f"{LETTERS[i]}) {o}" for i, o in enumerate(item.options)]
    if item.qtype == "fill_blank":
        return ["Fill in the blank(s): " + item.question]
    if item.qtype == "true_false":
        return ["State whether True or False: " + item.question]
    if item.qtype == "match":
        lines = [item.question, "Column A:"] + [f"{i + 1}. {t}" for i, t in enumerate(item.left)]
        return lines + ["Column B:"] + [f"{LETTERS[i]}. {d}" for i, d in enumerate(item.options)]
    return [item.question]


def answer_lines(item: QAItem) -> List[str]:
    """The answer in a readable form (keywords are shown separately by the UI)."""
    if item.qtype == "mcq":
        return [f"Correct option: {LETTERS[item.correct_index]}) {item.answer}"]
    if item.qtype == "true_false":
        lines = [f"Answer: {item.answer}"]
        if item.answer == "False":
            lines.append(f"Correct statement: {item.evidence}")
        return lines
    if item.qtype == "match":
        return [f"{i + 1}. {t}  →  {LETTERS[item.match_key[i]]}" for i, t in enumerate(item.left)]
    return [f"Answer: {item.answer}"]


def item_to_text(item: QAItem, number: int, with_answers: bool) -> str:
    """One question as plain text (for the download button)."""
    lines = question_lines(item)
    text = f"{number}. [{QUESTION_TYPES.get(item.qtype, item.qtype)} | {item.topic}] {lines[0]}"
    text += "".join("\n   " + l for l in lines[1:])
    if with_answers:
        text += "\n   " + "\n   ".join(answer_lines(item))
        text += "\n   Keywords: " + ", ".join(item.keywords)
    return text
