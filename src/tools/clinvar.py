"""Live ClinVar lookup, exposed as a discoverable tool.

Wraps `dnadet.tools.clinical`, which does the work: NCBI E-utilities lookup by
VariationID or by GRCh37 position, then verification that the record actually
describes this candidate - matching coordinate AND allele, because two ClinVar
records at one position can differ by a single base and attaching the wrong
one's classification would look entirely correct in the output.

Raw responses are cached under outputs/blockC/cache/, so a re-run costs no
network calls and the cache doubles as the retrieval audit trail.
"""

from __future__ import annotations

from src import bridge
from src.models import Candidate, Case, Evidence
from src.tools.registry import Tool

from dnadet.tools.clinical import process as _clinvar_process


def investigate(candidate: Candidate, case: Case) -> list[Evidence]:
    """Return ClinVar evidence for one candidate. Never raises on lookup failure."""
    record = bridge.to_engine_dict(candidate)
    try:
        rows, _drift = _clinvar_process([record], bridge.cache_dir(), replay=False,
                                        limit=1)
    except Exception as exc:  # noqa: BLE001 - a dead source is a gap, not a crash
        return bridge.to_model_evidence([{
            "candidate_id": candidate.candidate_id, "category": "clinvar",
            "source": "ClinVar (live) - lookup failed",
            "assembly": case.assembly,
            "interpretation": (
                f"ClinVar could not be reached ({type(exc).__name__}). This is a "
                "GAP, not a negative result - absence of a retrieved record is not "
                "evidence that no record exists."
            ),
            "limitations": ["Live retrieval failed; re-run before submission."],
        }], "C")
    # ClinGen rows come from the sibling tool; keep this one's output clean.
    return bridge.to_model_evidence(
        [r for r in rows if r.category != "clingen"], "C")


TOOL = Tool(
    name="clinvar",
    description="Live ClinVar classification, review status and submission counts, verified by locus and allele.",
    investigate=investigate,
)
