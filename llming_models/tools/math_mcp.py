"""Backwards-compatibility re-export — MathMCP has moved to llming-lodge.

The canonical location is ``llming_lodge.tools.math_mcp``.
This stub re-exports the class so existing ``from llming_models.tools.math_mcp import ...``
imports continue to work.
"""
from __future__ import annotations

try:
    from llming_lodge.tools.math_mcp import (  # type: ignore[import-untyped]
        MathMCP,
        _sympy_eval,
        _rnd,
        _expr_to_latex,
        _expr_to_str,
        _js_escape,
        _html_escape,
    )
except ImportError:
    raise ImportError(
        "MathMCP has moved to the llming-lodge package. "
        "Install it with: pip install llming-lodge"
    )

__all__ = ["MathMCP", "_sympy_eval", "_rnd", "_expr_to_latex", "_expr_to_str", "_js_escape", "_html_escape"]
