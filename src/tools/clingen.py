"""ClinGen expert-panel curation lookup, exposed as a discoverable tool.

ClinGen answers a different question from ClinVar. ClinVar: has anyone called
this VARIANT pathogenic? ClinGen ERepo: has an expert panel curated it, which is
the only 3-star evidence tier there is. An empty result is informative - it
means no Variant Curation Expert Panel has looked at the gene, so ClinVar
submitter consensus is the ceiling on clinical evidence for every variant in it.
"""

from __future__ import annotations

from src import bridge
from src.models import Candidate, Case, Evidence
from src.tools.registry import Tool

from dnadet.tools.clinical import CLINGEN_VALIDITY, clingen_gene_validity

_SEEN: set[str] = set()


def investigate(candidate: Candidate, case: Case) -> list[Evidence]:
    """Return ClinGen evidence for a candidate's gene. Queried once per gene."""
    gene = candidate.gene
    if not gene or gene in _SEEN:
        return []
    _SEEN.add(gene)

    try:
        body, _url, stamp = clingen_gene_validity(gene, bridge.cache_dir(),
                                                  replay=False)
    except Exception as exc:  # noqa: BLE001
        body, stamp = None, ""
        note = f"ClinGen unreachable ({type(exc).__name__})."
    else:
        note = ""

    if not isinstance(body, dict):
        return bridge.to_model_evidence([{
            "candidate_id": candidate.candidate_id, "category": "clingen",
            "source": "ClinGen ERepo - lookup failed", "record_or_accession": gene,
            "assembly": case.assembly, "retrieved_at": stamp,
            "interpretation": note or "No parseable response from ClinGen. GAP, "
                                      "not a negative result.",
            "limitations": ["Live retrieval failed; re-run before submission."],
        }], "C")

    interps = body.get("variantInterpretations") or []
    return bridge.to_model_evidence([{
        "candidate_id": candidate.candidate_id, "category": "clingen",
        "source": "ClinGen Evidence Repository (ERepo)",
        "record_or_accession": gene, "query": f"matchMode=exact&gene={gene}",
        "assembly": case.assembly, "raw_field": "variantInterpretations",
        "raw_value": f"{len(interps)} interpretation(s)",
        "tool_or_data_version": f"ClinGen ERepo, retrieved {stamp}",
        "url": f"{CLINGEN_VALIDITY}?search={gene}", "retrieved_at": stamp,
        "interpretation": (
            f"ClinGen expert panels have published {len(interps)} variant "
            f"interpretation(s) for {gene}."
            + (" No Variant Curation Expert Panel has curated this gene, so no "
               "3-star expert classification exists for any variant in it; "
               "ClinVar submitter consensus is the strongest tier available."
               if not interps else "")
        ),
        "limitations": [
            "ERepo covers VCEP-curated variants only; it is not a gene-disease "
            "validity classification.",
            "Absence of a curation says nothing about the variant - it says the "
            "gene has not been through expert curation.",
        ],
    }], "C")


TOOL = Tool(
    name="clingen",
    description="ClinGen Evidence Repository check for expert-panel (3-star) variant curation in the candidate's gene.",
    investigate=investigate,
)
