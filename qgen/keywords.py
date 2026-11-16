"""
keywords.py
-----------
Extracts the most informative keywords / key-phrases of an answer using TF-IDF,
computed against the sentences of the source passage.
"""
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .extractor import split_sentences

# generic verbs / fillers that TF-IDF would otherwise rank highly inside a short answer
EXTRA_STOPWORDS = ["use", "uses", "used", "using", "called", "known", "refers", "refer", "include",
                   "includes", "occurs", "occur", "result", "results", "process", "type", "types"]
STOPWORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))


def extract_keywords(answer: str, context: str, top_k: int = 6) -> List[str]:
    """
    Rank the words of `answer` by TF-IDF weight (IDF is learned from the sentences
    of `context`, so terms that are specific to this answer score higher than generic ones).

    Returns up to `top_k` distinct keywords.
    """
    if not answer.strip():
        return []

    docs = split_sentences(context) + [answer]
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:  # empty vocabulary (e.g. answer is only stop-words)
        return []

    scores = matrix[-1].toarray()[0]
    terms = vectorizer.get_feature_names_out()
    keywords: List[str] = []
    for i in scores.argsort()[::-1]:
        if scores[i] <= 0 or len(keywords) >= top_k:
            break
        term = terms[i]
        if term in keywords:
            continue
        keywords.append(term)
    return keywords
