"""Unit tests for the generic retrieval primitives (chunking + BM25)."""
from __future__ import annotations

from llming_models.retrieval import BM25Index, Segment, chunk_segments
from llming_models.utils.tokenization import count_tokens


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 1


def test_chunk_segments_overlap_and_keys():
    segs = [Segment(text=f"segment number {i} content words here", key=i) for i in range(1, 11)]
    chunks = chunk_segments(segs, chunk_tokens=20, overlap=5)
    assert chunks
    # Every segment key is covered by at least one chunk.
    covered = set()
    for c in chunks:
        covered.update(c.keys)
    assert covered == set(range(1, 11))
    # Keys within a chunk are ordered-unique.
    for c in chunks:
        assert c.keys == sorted(set(c.keys))
        assert c.token_count > 0


def test_chunk_segments_skips_empty():
    segs = [Segment(text="", key=1), Segment(text="real content here", key=2)]
    chunks = chunk_segments(segs)
    assert chunks and all(2 in c.keys for c in chunks)
    assert all(1 not in c.keys for c in chunks)


def test_bm25_ranks_and_handles_part_numbers():
    texts = [
        "the cat sat on the mat",
        "warranty clause part 445.123.017 applies to all orders",
        "pricing and delivery terms",
    ]
    idx = BM25Index(texts)
    hits = idx.search("445.123.017 warranty", top_k=3)
    assert hits, "expected a hit"
    assert hits[0][0] == 1  # the warranty/part-number doc ranks first
    # Unknown query returns nothing.
    assert idx.search("zzzzz nonexistent", top_k=3) == []


def test_bm25_empty_index():
    assert BM25Index([]).search("anything") == []
