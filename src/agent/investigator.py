"""Call selected evidence tools for a shortlist, then rank from the evidence."""

from __future__ import annotations

from src import bridge
from src.models import Candidate, Case, Evidence
from src.tools import ToolRegistry

from dnadet.agent import assess as _assess
from dnadet.submission import FOLLOW_UPS as _FOLLOW_UP_QUESTIONS
from dnadet.submission import confidence_for as _confidence_for
from dnadet.qa import answer as _qa_answer


def investigate_candidates(candidates: list[Candidate], case: Case, tool_names: list[str], registry: ToolRegistry) -> list[Evidence]:
    """Run only the explicitly selected tools and return their traceable evidence."""
    evidence: list[Evidence] = []
    for candidate in candidates:
        for tool_name in tool_names:
            evidence.extend(registry.get(tool_name).investigate(candidate, case))
    return evidence


def reason_over_evidence(candidates: list[Candidate], evidence: list[Evidence]) -> list[Candidate]:
    """Rank candidates by weighted, ORIGIN-COLLAPSED evidence strength.

    The ranking is not a count of evidence rows. Rows that trace to the same
    origin are one argument stated several times, not several arguments: a
    ClinVar classification, an ACMG criterion derived from it, and an Exomiser
    whitelist granted on the strength of it all reduce to one ClinVar record.
    Counting them separately would manufacture agreement out of a single source.

    Weights are ordinal, not calibrated - they encode only that expert-reviewed
    clinical evidence and real phenotype-term overlap outweigh in-silico
    prediction. `confidence` is a documented transform of that strength and is
    NOT a probability; there is one patient and no ground truth to calibrate on.
    """
    rows = bridge.evidence_as_dicts(evidence)
    assessments = [_assess(bridge.to_engine_dict(c), rows) for c in candidates]
    order = sorted(assessments, key=lambda a: a.sort_key)

    by_id = {c.candidate_id: c for c in candidates}
    ranked: list[Candidate] = []
    for rank, a in enumerate(order, 1):
        candidate = by_id[a.candidate_id]
        candidate.rank = rank
        candidate.supporting_evidence_ids = sorted({e for v in a.verdicts if v.stance == "supports" for e in v.evidence_ids})
        candidate.conflicting_evidence_ids = sorted({e for v in a.verdicts if v.stance == "conflicts" for e in v.evidence_ids})
        candidate.missing_evidence = a.gaps
        candidate.confidence = _confidence_for(a.strength)
        reason = a.reason
        if a.circularity:
            reason += " Circularity noted: " + " ".join(a.circularity)
        candidate.reason_for_rank = reason
        ranked.append(candidate)
    return ranked


def follow_up_examples(candidates: list[Candidate], evidence: list[Evidence], policy: str = "template", backend: str = "groq", model: str = "") -> list[dict[str, str]]:
    """Answer the brief's follow-up questions from the stored evidence only.

    Retrieval and comparison are deterministic. When `policy` is "llm" a model
    phrases the answer, receives nothing but the retrieved rows, and every
    evidence ID it emits is checked against the log before the answer is kept -
    a citation that looks real and is not would be the most damaging thing this
    system could output.
    """
    rows = bridge.evidence_as_dicts(evidence)
    ordered = sorted([_assess(bridge.to_engine_dict(c), rows) for c in candidates], key=lambda a: a.sort_key)
    return [{"user": question, "agent": _qa_answer(question, ordered, rows, policy, backend, model)} for question in _FOLLOW_UP_QUESTIONS]
