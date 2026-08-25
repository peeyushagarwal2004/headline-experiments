"""Interpretable headline features - the baseline a transformer has to beat.

These encode the folklore of viral headline writing: curiosity gaps, second
person, numerals, superlatives, emotional intensity. Cheap to compute and
fully interpretable, so a fitted coefficient is a readable statement about
what made Upworthy headlines win.

They also set the bar honestly. A fine-tuned transformer that cannot beat
seventeen hand-written regexes has not earned its GPU time, and reporting that
comparison is more useful than reporting the transformer's number alone.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SECOND_PERSON = re.compile(r"\b(you|your|you're|yours|yourself)\b", re.I)
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our|us)\b", re.I)
DEMONSTRATIVE = re.compile(r"\b(this|that|these|those)\b", re.I)
SUPERLATIVE = re.compile(r"\b(best|worst|most|least|ever|never|always|greatest|\w+est)\b", re.I)
CURIOSITY = re.compile(
    r"\b(what|why|how|when|who|secret|reason|actually|really|truth|"
    r"happened|happens|surprising|shocking|wait|watch|reveals?)\b",
    re.I,
)
EMOTION = re.compile(
    r"\b(amazing|incredible|heartbreaking|beautiful|powerful|perfect|"
    r"terrible|awful|hilarious|adorable|inspiring|brilliant|stunning)\b",
    re.I,
)
MODAL = re.compile(r"\b(will|won't|can't|cannot|should|must|need)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|don't|doesn't|isn't)\b", re.I)

FEATURE_NAMES = [
    "n_chars", "n_words", "mean_word_len", "n_sentences",
    "has_question", "has_exclamation", "has_quote", "has_colon", "has_ellipsis",
    "n_digits", "starts_with_digit",
    "second_person", "first_person", "demonstrative",
    "superlative", "curiosity", "emotion", "modal", "negation",
    "n_allcaps_words", "capital_ratio",
]


def featurise_one(text: str) -> list[float]:
    words = text.split()
    n_words = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]
    return [
        len(text),
        len(words),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        float(len(re.findall(r"[.!?]+", text)) or 1),
        float("?" in text),
        float("!" in text),
        float('"' in text or "'" in text),
        float(":" in text),
        float("..." in text),
        float(sum(c.isdigit() for c in text)),
        float(bool(re.match(r"^\s*\d", text))),
        len(SECOND_PERSON.findall(text)) / n_words,
        len(FIRST_PERSON.findall(text)) / n_words,
        len(DEMONSTRATIVE.findall(text)) / n_words,
        len(SUPERLATIVE.findall(text)) / n_words,
        len(CURIOSITY.findall(text)) / n_words,
        len(EMOTION.findall(text)) / n_words,
        len(MODAL.findall(text)) / n_words,
        len(NEGATION.findall(text)) / n_words,
        float(sum(w.isupper() and len(w) > 1 for w in words)),
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
    ]


def featurise(texts) -> np.ndarray:
    return np.asarray([featurise_one(str(t)) for t in texts], dtype=float)


def pair_feature_matrix(pairs: pd.DataFrame) -> np.ndarray:
    """Feature DIFFERENCE, b minus a.

    Differencing is what makes the representation antisymmetric: swapping the
    two headlines negates every feature, so a linear model with no intercept
    is guaranteed to give the mirrored prediction. Without that, a model can
    pick up on which slot a winner tends to occupy - an artefact of how the
    pairs were built, not a property of headlines.
    """
    return featurise(pairs["headline_b"]) - featurise(pairs["headline_a"])
