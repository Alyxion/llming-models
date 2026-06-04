"""Generic overlapping text chunker.

Splits a sequence of labelled segments into ~``chunk_tokens`` windows with
``overlap`` token overlap, recording which segment labels each window spans.
Labels are opaque (the large-document pipeline uses page numbers); this module
has no document/PDF knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..utils.tokenization import count_tokens

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 120


@dataclass
class Segment:
    """An input unit of text with an opaque label (e.g. a page number)."""

    text: str
    key: Any


@dataclass
class TextChunk:
    """An output window of text and the (ordered, unique) labels it spans."""

    text: str
    token_count: int
    keys: list[Any] = field(default_factory=list)


def _windows(n: int, chunk_tokens: int, overlap: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    approx_tokens_per_word = 1.3
    win = max(1, int(chunk_tokens / approx_tokens_per_word))
    step = max(1, int((chunk_tokens - overlap) / approx_tokens_per_word))
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + win)
        out.append((start, end))
        if end >= n:
            break
        start += step
    return out


def chunk_segments(
    segments: list[Segment],
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
    counter: Callable[[str], int] = count_tokens,
) -> list[TextChunk]:
    """Chunk labelled segments into overlapping windows."""
    words: list[tuple[str, Any]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        for w in text.split():
            words.append((w, seg.key))
    if not words:
        return []

    chunks: list[TextChunk] = []
    for s, e in _windows(len(words), chunk_tokens, overlap):
        span = words[s:e]
        if not span:
            continue
        text = " ".join(w for w, _ in span)
        # Ordered-unique keys spanned by this window.
        keys: list[Any] = []
        seen = set()
        for _, k in span:
            if k not in seen:
                seen.add(k)
                keys.append(k)
        chunks.append(TextChunk(text=text, token_count=counter(text), keys=keys))
    return chunks
