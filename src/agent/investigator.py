"""Call selected evidence tools for an already-created candidate shortlist."""

from __future__ import annotations

from src.models import Candidate, Case, Evidence
from src.tools import ToolRegistry


def investigate_candidates(candidates: list[Candidate], case: Case, tool_names: list[str], registry: ToolRegistry) -> list[Evidence]:
    """Run only the explicitly selected tools and return their traceable evidence."""
    evidence: list[Evidence] = []
    for candidate in candidates:
        for tool_name in tool_names:
            evidence.extend(registry.get(tool_name).investigate(candidate, case))
    return evidence


def reason_over_evidence(candidates: list[Candidate], evidence: list[Evidence]) -> list[Candidate]:
    """TODO: Compare evidence and assign transparent ranks without inventing claims."""
    del candidates, evidence
    raise NotImplementedError("Agent reasoning is intentionally unimplemented. Add it in src/agent/investigator.py.")

