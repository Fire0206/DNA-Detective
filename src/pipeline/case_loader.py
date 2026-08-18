"""Prepare a case from a VCF path and a phenotype file."""

from __future__ import annotations

import re
from pathlib import Path

from src.models import Case

HPO_ID = re.compile(r"^\s+id:\s+(HP:\d+)\s*$")
ASSEMBLY = re.compile(r"^\s+genomeAssembly:\s+(.+?)\s*$")


def load_case(vcf_path: Path, phenopacket_path: Path, case_id: str = "") -> Case:
    """Read lightweight case metadata without parsing or annotating VCF records."""
    if not vcf_path.is_file():
        raise FileNotFoundError(f"VCF was not found: {vcf_path}")
    if not phenopacket_path.is_file():
        raise FileNotFoundError(f"Phenopacket was not found: {phenopacket_path}")
    sample = ""
    with vcf_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                sample = columns[9] if len(columns) > 9 else ""
                break
    hpo_terms: list[str] = []
    assembly = ""
    for line in phenopacket_path.read_text(encoding="utf-8").splitlines():
        if match := HPO_ID.match(line):
            hpo_terms.append(match.group(1))
        elif match := ASSEMBLY.match(line):
            assembly = match.group(1)
    return Case(case_id=case_id or sample or vcf_path.stem, vcf_path=str(vcf_path), phenopacket_path=str(phenopacket_path), assembly=assembly, sample=sample, hpo_terms=hpo_terms)

