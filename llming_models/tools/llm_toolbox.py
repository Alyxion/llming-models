from __future__ import annotations

from typing import Any

from .llm_tool import LlmTool

class LlmToolbox:
    """
    A collection of tools that can be used by an LLM.
    """

    def __init__(self, name: str, description: str, tools: list[LlmTool | str | dict[str, Any]]) -> None:
        """
        Initialize the toolbox.
        """
        self.name = name
        self.description = description
        self.tools = tools
