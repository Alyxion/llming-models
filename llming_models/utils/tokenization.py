"""Canonical token-counting helper for the engine.

Uses tiktoken's ``cl100k_base`` (already a dependency) with a char/4 fallback so
it never hard-fails. This is the single place a token count should come from;
prefer it over re-importing ``get_encoding`` ad hoc.
"""
from __future__ import annotations

_ENCODER = None
_ENCODER_TRIED = False


def get_default_encoder():
    """Return the shared cl100k_base encoder, or None if unavailable."""
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            from tiktoken import get_encoding
            _ENCODER = get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Approximate token count for ``text``."""
    if not text:
        return 0
    enc = get_default_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)
