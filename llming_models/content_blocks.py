"""Fenced code block encode/decode utilities for content-embedded data.

Rich content (plots, tables, rendered MCP output, etc.) is stored as fenced
code blocks inside message content strings.  This module provides a generic
mechanism -- it does not know about specific block types.

Format::

    ```block_type
    {"key": "value", ...}
    ```
"""
from __future__ import annotations

import json
import re
from typing import Any

_BLOCK_RE = re.compile(
    r"```(\w+)\n(.*?)\n```",
    re.DOTALL,
)


def encode_content_block(block_type: str, data: dict[str, Any] | list[Any]) -> str:
    """Encode a data block as a fenced code block string.

    Returns a string like ``\\n\\n```block_type\\n{json}\\n``` `` ready to be
    appended to message content.
    """
    return f"\n\n```{block_type}\n{json.dumps(data, ensure_ascii=False)}\n```"


def extract_content_blocks(content: str) -> list[tuple[str, str]]:
    """Extract all fenced code blocks from content.

    Returns a list of ``(block_type, raw_json_string)`` tuples.
    The content string is not modified (non-destructive).
    """
    return [(m.group(1), m.group(2)) for m in _BLOCK_RE.finditer(content)]


def strip_content_blocks(content: str) -> str:
    """Remove all fenced code blocks from content, returning plain text."""
    return _BLOCK_RE.sub("", content).strip()
