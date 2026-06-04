"""Tiny pure-Python BM25 over a list of texts.

Generic and dependency-free. Operates on plain strings and returns
``(index, score)`` so callers can map hits back to their own objects. The
tokenizer keeps part-number-like tokens (``445.123.17``, ``abc-12``) and German
umlauts intact, which matters for catalog/spec corpora.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[0-9a-zäöüß]+(?:[.\-/][0-9a-zäöüß]+)*")

_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """In-memory BM25 over a fixed list of document texts."""

    def __init__(self, texts: list[str]) -> None:
        self._docs: list[list[str]] = [tokenize(t) for t in texts]
        self._lens = [len(d) for d in self._docs]
        self._avglen = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        self._tf: list[Counter] = [Counter(d) for d in self._docs]
        self._df: Counter = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        self._n = len(texts)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(index, score)`` pairs, best first."""
        if self._n == 0:
            return []
        q_terms = [t for t in tokenize(query) if t]
        if not q_terms:
            return []
        scored: list[tuple[int, float]] = []
        for i in range(self._n):
            tf = self._tf[i]
            dl = self._lens[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + _K1 * (1 - _B + _B * dl / (self._avglen or 1))
                score += self._idf(term) * (f * (_K1 + 1)) / denom
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
