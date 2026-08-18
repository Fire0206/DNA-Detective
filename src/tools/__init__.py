"""External evidence tool interfaces and automatic discovery helpers."""

from __future__ import annotations

import importlib
import pkgutil

from .registry import Tool, ToolRegistry


def discover_tools() -> ToolRegistry:
    """Load every module exposing ``TOOL = Tool(...)`` in this package.

    A collaborator can add ``src/tools/my_source.py`` with one ``TOOL`` object;
    no central registry file needs editing.
    """
    registry = ToolRegistry()
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_") or module_info.name == "registry":
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        tool = getattr(module, "TOOL", None)
        if tool is not None:
            if not isinstance(tool, Tool):
                raise TypeError(f"{module.__name__}.TOOL must be a Tool instance")
            registry.register(tool)
    return registry


__all__ = ["Tool", "ToolRegistry", "discover_tools"]

