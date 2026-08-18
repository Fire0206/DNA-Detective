"""Candidate filtering and prioritization extension point."""

from __future__ import annotations

from src.models import Candidate, Case


def generate_candidate_shortlist(case: Case, limit: int = 10) -> list[Candidate]:
    """Create a ranked candidate shortlist.

    TODO: Implement staged VCF filtering, annotation, and phenotype-aware ranking.
    Preserve filter reasons and avoid treating this placeholder as clinical analysis.
    """
    del case, limit
    raise NotImplementedError("Candidate prioritization is intentionally unimplemented. Add it in src/pipeline/prioritization.py.")

