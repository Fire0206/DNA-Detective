"""Conversion layer between the `dnadet` engine and this repo's `src.models`.

Two packages describe the same submission contract. `src.models` owns the
OUTPUT shape - it mirrors starter/submission_template.json and is the authority
on what gets written. `dnadet.contract` owns the EVIDENCE shape used while
reasoning, which carries working fields the template has no slot for (the
snapshot ClinVar VariationID, the filter trail, the Exomiser phenotype-score
provenance).

Rather than pick one and lose information, this module converts at the boundary
and keeps the engine's richer record in a side table so tools can still reach
the fields the template drops. `dnadet` never imports `src`; the dependency runs
one way only, so the engine stays usable and testable on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.models import Candidate, Evidence, PhenotypeMatch, Variant

# candidate_id -> the engine's full record. Populated by the shortlist stage;
# read by the tool wrappers, which need fields `src.models.Candidate` omits.
_ENGINE_RECORDS: dict[str, dict[str, Any]] = {}

# Evidence IDs must be unique across every tool call in a run. Each tool call
# numbers its own rows from 1, so they are renumbered here on arrival.
_EVIDENCE_SEQ = {"n": 0}


def reset() -> None:
    _ENGINE_RECORDS.clear()
    _EVIDENCE_SEQ["n"] = 0


def remember(record: dict[str, Any]) -> None:
    _ENGINE_RECORDS[record["candidate_id"]] = record


def engine_record(candidate_id: str) -> dict[str, Any]:
    """The engine's record for a candidate, or a minimal one built from the model."""
    return _ENGINE_RECORDS.get(candidate_id, {})


def engine_records() -> list[dict[str, Any]]:
    return list(_ENGINE_RECORDS.values())


def to_engine_dict(candidate: Candidate) -> dict[str, Any]:
    """A `src.models.Candidate` as the engine expects it, enriched from the side table."""
    v = candidate.normalized_variant
    base = {
        "candidate_id": candidate.candidate_id,
        "chrom": v.chrom, "pos": v.pos, "ref": v.ref, "alt": v.alt,
        "gene": candidate.gene, "transcript": candidate.transcript,
        "consequence": candidate.consequence, "zygosity": candidate.zygosity,
        "hgvs_c": v.hgvs_c, "hgvs_p": v.hgvs_p,
        "missing_evidence": list(candidate.missing_evidence),
    }
    # Engine-only fields (clinvar_vcv, effect_scores, filter_trail, ...) survive
    # only through the side table - the template has nowhere to put them.
    merged = dict(engine_record(candidate.candidate_id))
    merged.update({k: val for k, val in base.items() if val not in (None, "", [])})
    return merged


def to_model_candidate(record: dict[str, Any], rank: int | None = None) -> Candidate:
    """Engine record -> the dataclass that gets serialised into the report."""
    pm = record.get("phenotype_match") or {}
    return Candidate(
        candidate_id=record["candidate_id"],
        normalized_variant=Variant(
            assembly=record.get("assembly", "GRCh37"),
            chrom=str(record["chrom"]), pos=int(record["pos"]),
            ref=record["ref"], alt=record["alt"],
            hgvs_g=record.get("hgvs_g", "") or "",
            hgvs_c=record.get("hgvs_c") or "",
            hgvs_p=record.get("hgvs_p") or "",
        ),
        rank=rank,
        gene=record.get("gene") or "",
        transcript=record.get("transcript") or "",
        consequence=record.get("consequence") or "",
        zygosity=record.get("zygosity") or "",
        candidate_disease=record.get("candidate_disease", "") or "",
        inheritance=record.get("inheritance", "") or "",
        phenotype_match=PhenotypeMatch(
            score=pm.get("score"),
            matched_hpo_terms=list(pm.get("matched_hpo_terms", [])),
            explanation=pm.get("explanation", "") or "",
        ),
        missing_evidence=list(record.get("missing_evidence", [])),
        reason_for_rank=record.get("reason_for_rank", "") or "",
        confidence=record.get("confidence"),
    )


def to_model_evidence(rows: Iterable[Any], prefix: str) -> list[Evidence]:
    """Engine Evidence (dataclass or dict) -> `src.models.Evidence`, renumbered."""
    out: list[Evidence] = []
    for row in rows:
        d = row if isinstance(row, dict) else row.to_dict()
        _EVIDENCE_SEQ["n"] += 1
        out.append(Evidence(
            evidence_id=f"{prefix}{_EVIDENCE_SEQ['n']:03d}",
            candidate_id=d.get("candidate_id", ""),
            category=d.get("category", ""),
            source=d.get("source", ""),
            record_or_accession=d.get("record_or_accession", "") or "",
            query=d.get("query", "") or "",
            assembly=d.get("assembly", "GRCh37") or "GRCh37",
            transcript=d.get("transcript"),
            raw_field=d.get("raw_field", "") or "",
            raw_value=d.get("raw_value", "") or "",
            tool_or_data_version=d.get("tool_or_data_version", "") or "",
            url=d.get("url", "") or "",
            record_kind=d.get("record_kind", "") or "",
            retrieved_at=d.get("retrieved_at", "") or "",
            interpretation=d.get("interpretation", "") or "",
            limitations=list(d.get("limitations", [])),
        ))
    return out


def evidence_as_dicts(evidence: Iterable[Evidence]) -> list[dict[str, Any]]:
    """`src.models.Evidence` -> plain dicts, which is what the engine reads."""
    return [{
        "evidence_id": e.evidence_id, "candidate_id": e.candidate_id,
        "category": e.category, "source": e.source,
        "record_or_accession": e.record_or_accession, "query": e.query,
        "assembly": e.assembly, "transcript": e.transcript,
        "raw_field": e.raw_field, "raw_value": e.raw_value,
        "tool_or_data_version": e.tool_or_data_version, "url": e.url,
        "record_kind": getattr(e, "record_kind", ""),
        "retrieved_at": e.retrieved_at, "interpretation": e.interpretation,
        "limitations": list(e.limitations),
    } for e in evidence]


def cache_dir(base: str = "outputs/blockC") -> str:
    Path(base).mkdir(parents=True, exist_ok=True)
    return base
