"""Generic, LLM-agnostic retrieval primitives.

Building blocks for retrieval-augmented context — text chunking and a small
pure-Python BM25 — with no knowledge of documents, PDFs, or the chat server.
Hosts (e.g. llming-lodge's large-document pipeline) build domain objects on top
of these. Token counting comes from ``llming_models.utils.tokenization`` so the
engine has a single tokenizer.
"""
from __future__ import annotations

from .bm25 import BM25Index, tokenize
from .chunker import CHUNK_OVERLAP, CHUNK_TOKENS, Segment, TextChunk, chunk_segments

__all__ = [
    "BM25Index",
    "tokenize",
    "Segment",
    "TextChunk",
    "chunk_segments",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP",
]
