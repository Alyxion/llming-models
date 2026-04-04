from __future__ import annotations

from collections.abc import Callable
from typing import Any

class LlmTool:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}
