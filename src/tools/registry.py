"""Tiny registry so a new tool can be added without changing the pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.models import Candidate, Case, Evidence

ToolFunction = Callable[[Candidate, Case], list[Evidence]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    investigate: ToolFunction


class ToolRegistry:
    """Register named candidate-investigation functions."""
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Unknown tool: {name}") from error

    def names(self) -> list[str]:
        return sorted(self._tools)

