"""Small, serializable models compatible with starter/submission_template.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Variant:
    """A normalized genomic variant; HGVS fields stay empty until annotated."""
    assembly: str
    chrom: str
    pos: int
    ref: str
    alt: str
    hgvs_g: str = ""
    hgvs_c: str = ""
    hgvs_p: str = ""


@dataclass
class PhenotypeMatch:
    score: float | None = None
    matched_hpo_terms: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class Candidate:
    """One candidate allele and the fields expected in the final submission."""
    candidate_id: str
    normalized_variant: Variant
    rank: int | None = None
    gene: str = ""
    transcript: str = ""
    consequence: str = ""
    zygosity: str = ""
    candidate_disease: str = ""
    inheritance: str = ""
    phenotype_match: PhenotypeMatch = field(default_factory=PhenotypeMatch)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    conflicting_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    reason_for_rank: str = ""
    confidence: float | None = None


@dataclass
class Evidence:
    """Traceable evidence returned by a tool or added by investigator logic."""
    evidence_id: str
    candidate_id: str
    category: str
    source: str
    record_or_accession: str = ""
    query: str = ""
    assembly: str = ""
    transcript: str | None = None
    raw_field: str = ""
    raw_value: str = ""
    tool_or_data_version: str = ""
    url: str = ""
    retrieved_at: str = ""
    interpretation: str = ""
    limitations: list[str] = field(default_factory=list)


@dataclass
class Case:
    """Prepared inputs shared by the filtering, tool, and reasoning stages."""
    case_id: str
    vcf_path: str
    phenopacket_path: str
    assembly: str
    sample: str
    hpo_terms: list[str]


@dataclass
class FinalReport:
    """The top-level report mirrors the starter submission template."""
    team: str
    case: Case
    agent_name: str = "DNA Detective"
    agent_version: str = "0.1.0"
    tools: list[str] = field(default_factory=list)
    run_started_at: str = ""
    run_finished_at: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    evidence_log: list[Evidence] = field(default_factory=list)
    follow_up_examples: list[dict[str, str]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready keys compatible with starter/submission_template.json."""
        return {
            "team": self.team,
            "case_id": self.case.case_id,
            "input": {"vcf": self.case.vcf_path, "phenopacket": self.case.phenopacket_path, "assembly": self.case.assembly, "sample": self.case.sample, "hpo_terms": self.case.hpo_terms},
            "method": {"agent_name": self.agent_name, "agent_version": self.agent_version, "tools": self.tools, "run_started_at": self.run_started_at, "run_finished_at": self.run_finished_at},
            "top_candidates": [asdict(candidate) for candidate in self.candidates],
            "evidence_log": [asdict(evidence) for evidence in self.evidence_log],
            "follow_up_examples": self.follow_up_examples,
            "limitations": self.limitations,
        }

