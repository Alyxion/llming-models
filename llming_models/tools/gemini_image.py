"""Backwards-compatibility re-export — gemini_image has moved to llming-lodge.

The canonical location is ``llming_lodge.tools.gemini_image``.
This stub re-exports the functions so existing imports continue to work.
"""
from __future__ import annotations

try:
    from llming_lodge.tools.gemini_image import (  # type: ignore[import-untyped]
        generate_image_with_gemini,
        generate_image_with_gemini_sync,
        generate_image_gemini_sync,
        get_google_api_key,
    )
except ImportError:
    raise ImportError(
        "gemini_image has moved to the llming-lodge package. "
        "Install it with: pip install llming-lodge"
    )

__all__ = [
    "generate_image_with_gemini",
    "generate_image_with_gemini_sync",
    "generate_image_gemini_sync",
    "get_google_api_key",
]
