"""Candidate filtering and prioritization.

Exomiser performs the annotation and the cascade (37,709 records -> 282 in 242
genes). This stage reads that output and turns it into candidates carrying a
traceable evidence row per source, plus a drop record with a stated reason for
every variant removed at the shortlist step.

It does NOT re-rank on Exomiser's combinedScore. That score is reproducible only
with the ACMG posterior as a third input, so its derivation cannot be shown from
the two published components; it orders the deep-dive queue and is never cited
as evidence. Ranking happens in src/agent/investigator.py, from evidence.

Requires an Exomiser run. Default location: data/exomiser/Pfeiffer-exomiser.jsonl
(under data/, not outputs/, because it is an INPUT to this stage and outputs/ is
gitignored). Override with DNA_DETECTIVE_EXOMISER_JSONL.
"""

from __future__ import annotations

import os
from pathlib import Path

from src import bridge
from src.models import Candidate, Case

from dnadet.exomiser_parse import build as _build_from_exomiser
from dnadet.exomiser_parse import load_records as _load_exomiser

DEFAULT_JSONL = "data/exomiser/Pfeiffer-exomiser.jsonl"
FALLBACK_JSONL = "outputs/exomiser/Pfeiffer-exomiser.jsonl"

_CACHE: dict[str, list] = {"evidence": [], "drops": []}


def _find_exomiser_output() -> Path:
    for candidate in (os.environ.get("DNA_DETECTIVE_EXOMISER_JSONL"), DEFAULT_JSONL, FALLBACK_JSONL):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No Exomiser output found. Expected one of:\n"
        f"  {DEFAULT_JSONL}\n  {FALLBACK_JSONL}\n"
        "or set DNA_DETECTIVE_EXOMISER_JSONL. Produce it with:\n"
        "  java -jar exomiser-cli-15.1.0.jar analyse --sample data/pfeiffer-phenopacket.yml --vcf data/Pfeiffer.vcf --assembly hg19"
    )


def generate_candidate_shortlist(case: Case, limit: int = 10) -> list[Candidate]:
    """Create a candidate shortlist with evidence and stated drop reasons."""
    del case  # assembly and HPO terms are fixed by the frozen contract
    path = _find_exomiser_output()
    records = _load_exomiser(str(path))

    # `build` asserts the spiked GENE=/INHERITANCE=/MIM= INFO fields never
    # reached the parsed records, and raises rather than silently using them.
    engine_candidates, engine_evidence, drops = _build_from_exomiser(records, top_genes=limit, max_variants_per_gene=3)

    bridge.reset()
    for record in engine_candidates:
        bridge.remember(record.to_dict())

    _CACHE["evidence"] = bridge.to_model_evidence(engine_evidence, "E")
    _CACHE["drops"] = [d.to_dict() for d in drops]
    return [bridge.to_model_candidate(r.to_dict()) for r in engine_candidates]


def shortlist_evidence() -> list:
    """Evidence rows produced while building the shortlist."""
    return list(_CACHE["evidence"])


def shortlist_drops() -> list[dict]:
    """One record per removed variant, each with the reason it was removed."""
    return list(_CACHE["drops"])
