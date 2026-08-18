"""Shared data contract for DNA Detective.

FREEZE THIS FILE FIRST. Every other module imports from here.
Owner: agreed by whole team before any other code is written.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

ASSEMBLY = "GRCh37"
SAMPLE = "manuel"

HPO_TERMS = [
    ("HP:0001159", "Syndactyly"),
    ("HP:0000486", "Strabismus"),
    ("HP:0000327", "Hypoplasia of the maxilla"),
    ("HP:0000520", "Proptosis"),
    ("HP:0000316", "Hypertelorism"),
    ("HP:0000244", "Brachyturricephaly"),
]

# INFO/FORMAT keys that leak the spiked-in answer. Never read these.
# See docs: exactly one record in Pfeiffer.vcf carries GENE=/INHERITANCE=/MIM=
# and a phased GT. Using them would bypass the entire task.
BANNED_FIELDS = {"GENE", "INHERITANCE", "MIM"}
BANNED_FORMAT = {"GT:DS:GL"}


def candidate_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Stable identifier used as the join key across every module."""
    return f"{chrom}-{pos}-{ref}-{alt}"


@dataclass
class Candidate:
    """One candidate variant. Produced by prefilter, enriched by tool layer."""

    candidate_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    rsid: Optional[str] = None          # dbSNP ID from VCF ID column, or None
    zygosity: str = ""                  # het | hom | other
    qual: Optional[float] = None
    depth: Optional[int] = None
    var_type: str = ""                  # snv | indel | mnv
    tier: str = ""                      # A | B | C  (prefilter priority)
    filter_trail: list[str] = field(default_factory=list)

    # --- filled in by the tool layer ---
    gene: Optional[str] = None
    transcript: Optional[str] = None
    consequence: Optional[str] = None
    hgvs_c: Optional[str] = None
    hgvs_p: Optional[str] = None
    gnomad_af: Optional[float] = None
    clinvar_vcv: Optional[str] = None
    clinvar_classification: Optional[str] = None
    clinvar_review_stars: Optional[int] = None
    effect_scores: dict[str, float] = field(default_factory=dict)
    phenotype_match: Optional[dict[str, Any]] = None

    # --- filled in by the agent loop ---
    supporting_evidence_ids: list[str] = field(default_factory=list)
    conflicting_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    reason_for_rank: str = ""
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    """One retrieval from one external source. Append-only; never overwrite."""

    evidence_id: str                    # E01, E02, ...
    candidate_id: str
    category: str                       # clinvar|clingen|gnomad|vep|effect|phenotype|literature
    source: str
    record_or_accession: str = ""
    query: str = ""
    assembly: str = ASSEMBLY
    transcript: Optional[str] = None
    raw_field: str = ""
    raw_value: str = ""
    tool_or_data_version: str = ""
    url: str = ""
    retrieved_at: str = ""              # ISO8601 Z
    interpretation: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DropRecord:
    """Why a variant left the candidate pool. Required by the brief."""

    candidate_id: str
    stage: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)
