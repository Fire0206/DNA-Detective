"""Block B - turn the Exomiser run into Candidates, Evidence and DropRecords.

Exomiser has already done the annotation and the narrowing (37,709 -> 282).
This module does NOT re-rank and does NOT trust `combinedScore` as an answer.
It does three things:

  1. builds a shortlist of Candidate objects with stable IDs;
  2. harvests the evidence that is ALREADY inside the JSONL - ClinVar accession,
     effect predictors, population-frequency absence, transcript consequence,
     phenotype match, ACMG criteria - each one provenance-tagged;
  3. records a DropRecord with a reason for every variant Exomiser removed and
     every gene that fell below the shortlist cutoff (README item 3).

Two rules that shape the whole file
-----------------------------------
MISSING vs ZERO.  A field that was never returned and a field returned as 0.0
mean different things: *no evidence* vs *evidence of absence*.  Nothing here
uses `dict.get(key, 0)`.  Absent stays None and is reported in
`missing_evidence`.  (The 2406 field is named `priorityScore` at the top level
and `phenotypeScore` inside `geneScores` - a `.get("phenotypeScore", 0)` on the
wrong level silently returns a believable 0.000 for every gene.)

NO LEAKED ANSWER.  One VCF record carries GENE=/INHERITANCE=/MIM= and a phased
genotype.  `assert_no_leak()` scans every parsed record and raises if those keys
survived into the Exomiser output.  We reach FGFR2 through evidence or not at all.

Usage
-----
    python3 -m dnadet.exomiser_parse --inspect outputs/exomiser/Pfeiffer-exomiser.jsonl
    python3 -m dnadet.exomiser_parse outputs/exomiser/Pfeiffer-exomiser.jsonl \
        --outdir outputs/blockB --top-genes 10 --max-variants-per-gene 3

Run `--inspect` FIRST.  Exomiser's JSON key names drift between releases; inspect
prints the real key paths in your file so a mismatch is loud instead of silent.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from .contract import (
    BANNED_FIELDS,
    BANNED_FORMAT,
    HPO_TERMS,
    Candidate,
    DropRecord,
    Evidence,
    candidate_id,
)

# --------------------------------------------------------------------------- #
# Provenance - every Evidence row carries this.  Change here, changes everywhere.
# --------------------------------------------------------------------------- #

EXOMISER_VERSION = "Exomiser 15.1.0"
EXOMISER_DATA_VERSION = "2406"
ASSEMBLY = "GRCh37/hg19"
PROVENANCE = f"{EXOMISER_VERSION} / data {EXOMISER_DATA_VERSION} / {ASSEMBLY}"

# The patient's six HPO terms. Declared once, in the frozen contract.
PATIENT_HPO = dict(HPO_TERMS)

# Exomiser's own frequency cutoff for this run, from the HTML filter summary.
# Frequencies in this file are PERCENTAGES.
COMMON_AF_PERCENT = 2.0

# Exomiser's default phenotype score when a gene has no known disease link.
# It is a placeholder, not a measurement.
NO_GENE_DISEASE_LINK = 0.5

# ClinVar review status -> star rating (ClinVar's own scale).
REVIEW_STARS = {
    "NO_ASSERTION_PROVIDED": 0,
    "NO_ASSERTION_CRITERIA_PROVIDED": 0,
    "NO_INTERPRETATION_FOR_SINGLE_VARIANT": 0,
    "CRITERIA_PROVIDED_SINGLE_SUBMITTER": 1,
    "CRITERIA_PROVIDED_CONFLICTING_INTERPRETATIONS": 1,
    "CRITERIA_PROVIDED_CONFLICTING_CLASSIFICATIONS": 1,
    "CRITERIA_PROVIDED_MULTIPLE_SUBMITTERS_NO_CONFLICTS": 2,
    "REVIEWED_BY_EXPERT_PANEL": 3,
    "PRACTICE_GUIDELINE": 4,
}

_SNAPSHOT_CAVEAT = (
    f"Offline Exomiser {EXOMISER_DATA_VERSION} snapshot (June 2024), not a live query. "
    "Re-check against the live source before citing; classification drift is possible."
)


class MissingType:
    """Sentinel. Distinct from None, 0, and '' so absence is never mistaken for a value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING = MissingType()


def dig(obj: Any, *path: Any) -> Any:
    """Walk nested dicts/lists. Returns MISSING - never a default value."""
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or not (-len(cur) <= key < len(cur)):
                return MISSING
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return MISSING
            cur = cur[key]
    return cur


def first_present(obj: Any, paths: Iterable[tuple], ) -> tuple[Any, Optional[tuple]]:
    """Try several key paths, return (value, path_that_worked).

    Exomiser renames fields between releases (priorityScore / phenotypeScore).
    This records WHICH path supplied the value so the evidence log can say so.
    """
    for path in paths:
        val = dig(obj, *path)
        if val is not MISSING:
            return val, path
    return MISSING, None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_records(path: str) -> list[dict]:
    """Read Exomiser output. Accepts JSON-lines or a single JSON array."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        raise ValueError(f"{path} is empty")

    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array of gene records")
        return data

    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {lineno}: {exc}") from exc
    return records


def assert_no_leak(records: list[dict]) -> None:
    """Fail loudly if the spiked INFO fields survived into the Exomiser output.

    Guards against reading GENE=/INHERITANCE=/MIM= or the GT:DS:GL format string,
    which would bypass the entire task.  Runs before any candidate is built.
    """
    banned_keys = {k.upper() for k in BANNED_FIELDS}

    def walk(node: Any, trail: str) -> Iterator[str]:
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).upper() in banned_keys:
                    yield f"{trail}.{k}"
                yield from walk(v, f"{trail}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{trail}[{i}]")
        elif isinstance(node, str) and node in BANNED_FORMAT:
            yield f"{trail} == {node!r}"

    hits = [h for i, rec in enumerate(records) for h in walk(rec, f"record[{i}]")]
    if hits:
        raise RuntimeError(
            "Leaked answer fields found in Exomiser output - refusing to continue.\n"
            + "\n".join(hits[:20])
        )


# --------------------------------------------------------------------------- #
# Field extraction (defensive - key names vary by Exomiser release)
# --------------------------------------------------------------------------- #


def norm_chrom(value: Any) -> str:
    s = str(value)
    return s[3:] if s.lower().startswith("chr") else s


def gene_symbol(rec: dict) -> str:
    val, _ = first_present(rec, [("geneSymbol",), ("geneIdentifier", "geneSymbol")])
    return "" if val is MISSING else str(val)


def gene_phenotype_score(rec: dict) -> tuple[Any, Optional[tuple]]:
    """The phenotype-match score, with the key path that supplied it.

    Top level calls it `priorityScore`; `geneScores[i]` calls it `phenotypeScore`.
    Never defaults to 0 - see the MISSING vs ZERO note at the top of the file.
    """
    return first_present(
        rec,
        [
            ("priorityScore",),
            ("phenotypeScore",),
            ("geneScores", 0, "phenotypeScore"),
            ("priorityResults", "HIPHIVE_PRIORITY", "score"),
        ],
    )


# FINDING - this run writes NO transcriptAnnotations block at all. The only
# consequence and HGVS strings in the file sit inside `clinVarData`, so they
# exist ONLY for variants that already carry a ClinVar record. A variant with no
# ClinVar entry has no consequence annotation anywhere in this output. That is a
# genuine gap in the offline evidence, not a parsing failure - and it is what
# makes a live Ensembl VEP call mandatory rather than optional for Block D.
#
# Consequence sourced this way is ClinVar's annotation, so it is NOT independent
# of the ClinVar classification. Do not present it as separate corroboration.
PLACEHOLDER_EFFECTS = {"SEQUENCE_VARIANT", ""}


def variant_effect(ve: dict) -> Any:
    """Molecular consequence, or MISSING. ClinVar-derived, never VEP-derived."""
    val, _ = first_present(
        ve,
        [
            ("variantEffect",),
            ("pathogenicityData", "clinVarData", "variantEffect"),
            ("transcriptAnnotations", 0, "variantEffect"),
        ],
    )
    if val is MISSING:
        return MISSING
    if (not clinvar_block(ve).get("variationId")
            and str(val).upper() in PLACEHOLDER_EFFECTS):
        # Exomiser's filler for an empty ClinVar record - not an annotation.
        return MISSING
    return val


def best_transcript(ve: dict) -> dict:
    """Transcript-level annotation. No accession is available in this run."""
    ann = dig(ve, "transcriptAnnotations", 0)
    if isinstance(ann, dict):
        return ann
    cv = clinvar_block(ve)
    if not cv:
        return {}
    return {
        "hgvsCdna": cv.get("hgvsCdna") or None,
        "hgvsProtein": cv.get("hgvsProtein") or None,
    }


def clinvar_block(ve: dict) -> dict:
    cv, _ = first_present(
        ve,
        [
            ("pathogenicityData", "clinVarData"),
            ("clinVarData",),
            ("pathogenicityData", "clinVar"),
        ],
    )
    return cv if isinstance(cv, dict) else {}


def predictor_scores(ve: dict) -> dict[str, float]:
    """{'REVEL': 0.965, ...}. Only sources actually present are included."""
    scores: dict[str, float] = {}
    raw, _ = first_present(
        ve,
        [
            # 15.1.0 / 2406 uses this name; the other two are older releases.
            ("pathogenicityData", "pathogenicityScores"),
            ("pathogenicityData", "predictedPathogenicityScores"),
            ("predictedPathogenicityScores",),
        ],
    )
    if isinstance(raw, list):
        for item in raw:
            src = dig(item, "source")
            val = dig(item, "score")
            if src is not MISSING and isinstance(val, (int, float)):
                scores[str(src)] = float(val)
    return scores


def frequency_state(ve: dict) -> tuple[str, Optional[float], str]:
    """Return (state, af_percent, raw). state is 'absent' | 'present' | 'not_queried'.

    'absent'      - the frequency sources were consulted and returned nothing.
    'not_queried' - no frequencyData block at all.  NOT the same thing.

    The population array is `frequencies` in 15.1.0 / 2406.  `empty: true` is
    emitted ONLY when the variant is absent; a variant with real frequencies has
    no `empty` key at all, so the two tests are complementary, not redundant.
    Frequencies are percentages, not fractions.
    """
    fd, _ = first_present(ve, [("frequencyData",), ("frequency",)])
    if fd is MISSING or not isinstance(fd, dict):
        return "not_queried", None, ""
    if fd.get("empty") is True:
        return "absent", None, json.dumps(fd, sort_keys=True)

    known, _ = first_present(fd, [("frequencies",), ("knownFrequencies",)])
    if isinstance(known, list) and known:
        pops = [
            (str(dig(k, "source")), float(dig(k, "frequency")))
            for k in known
            if isinstance(dig(k, "frequency"), (int, float))
        ]
        if pops:
            gnomad = [f for s, f in pops if s.upper().startswith("GNOMAD")]
            af = max(gnomad) if gnomad else max(f for _, f in pops)
            return "present", af, json.dumps(known, sort_keys=True)[:600]
    return "absent", None, json.dumps(fd, sort_keys=True)[:400]


def zygosity_of(ve: dict) -> str:
    gt, _ = first_present(ve, [("sampleGenotypes", 0, "call"), ("sampleGenotype",)])
    if gt is MISSING:
        sg = dig(ve, "sampleGenotypes")
        if isinstance(sg, dict) and sg:
            gt = next(iter(sg.values()))
    if not isinstance(gt, str):
        return ""
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) != 2:
        return "other"
    if alleles[0] == alleles[1] and alleles[0] not in ("0", "."):
        return "hom"
    if "0" in alleles:
        return "het"
    return "other"


def var_type_of(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "snv"
    return "indel"


# --------------------------------------------------------------------------- #
# Evidence harvesting
# --------------------------------------------------------------------------- #


class EvidenceLog:
    """Append-only. Hands out sequential IDs so every claim is citable as E007."""

    def __init__(self) -> None:
        self._rows: list[Evidence] = []

    def add(self, **kwargs: Any) -> Evidence:
        ev = Evidence(
            evidence_id=f"E{len(self._rows) + 1:03d}",
            retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **kwargs,
        )
        self._rows.append(ev)
        return ev

    @property
    def rows(self) -> list[Evidence]:
        return list(self._rows)


def harvest_evidence(
    log: EvidenceLog, cand: Candidate, gene_rec: dict, ve: dict, gene_score_rec: dict
) -> None:
    """One Evidence row per source. Attaches IDs back onto the Candidate."""
    cid = cand.candidate_id
    tx = best_transcript(ve)
    accession = tx.get("accession") if isinstance(tx, dict) else None

    def attach(ev: Evidence, supporting: bool = True) -> None:
        (cand.supporting_evidence_ids if supporting
         else cand.conflicting_evidence_ids).append(ev.evidence_id)

    # --- consequence / transcript ------------------------------------------
    eff = variant_effect(ve)
    if eff is not MISSING:
        attach(log.add(
            candidate_id=cid,
            category="vep",
            source="ClinVar record (via Exomiser 2406 snapshot)",
            record_or_accession=str(accession or ""),
            query=f"{cand.chrom}:{cand.pos} {cand.ref}>{cand.alt}",
            transcript=str(accession) if accession else None,
            raw_field="pathogenicityData.clinVarData.variantEffect",
            raw_value=str(eff),
            tool_or_data_version=PROVENANCE,
            interpretation=f"Molecular consequence: {eff}.",
            limitations=[
                "No transcript accession in this output - MANE Select unconfirmed.",
                "Annotation comes from the ClinVar record itself, so it is NOT "
                "independent of the ClinVar classification cited separately.",
                "Ensembl VEP required for an independent consequence call.",
            ],
        ))
    else:
        cand.missing_evidence.append(
            "no consequence annotation in Exomiser output (no transcriptAnnotations "
            "block, no ClinVar record) - requires live VEP"
        )

    # --- ClinVar ------------------------------------------------------------
    cv = clinvar_block(ve)
    variation_id = cv.get("variationId") or cv.get("alleleId")
    if variation_id:
        review = str(cv.get("reviewStatus", ""))
        interp = str(cv.get("primaryInterpretation", ""))
        cand.clinvar_vcv = str(variation_id)
        cand.clinvar_classification = interp or None
        cand.clinvar_review_stars = REVIEW_STARS.get(review.upper())
        conflicted = "CONFLICTING" in review.upper()
        # The submission split is the substance of a conflict, not the label.
        counts = cv.get("conflictingInterpretationCounts")
        counts_str = (
            "; ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            if isinstance(counts, dict) and counts else ""
        )
        attach(log.add(
            candidate_id=cid,
            category="clinvar",
            source="ClinVar (via Exomiser 2406 snapshot)",
            record_or_accession=f"VariationID {variation_id}",
            query=f"{cand.chrom}:{cand.pos} {cand.ref}>{cand.alt}",
            transcript=str(accession) if accession else None,
            raw_field="primaryInterpretation | reviewStatus | conflictingInterpretationCounts",
            raw_value=f"{interp} | {review}" + (f" | {counts_str}" if counts_str else ""),
            tool_or_data_version=PROVENANCE,
            url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/",
            interpretation=(
                f"ClinVar reports {interp or 'no classification'}; review status "
                f"{review or 'unknown'} "
                f"({REVIEW_STARS.get(review.upper(), '?')} star)."
                + (f" Submission split: {counts_str}." if counts_str else "")
            ),
            limitations=[_SNAPSHOT_CAVEAT] + (
                ["Submissions conflict - do not collapse to a single label."]
                if conflicted else []
            ) + (
                ["Exomiser whitelisted this variant on the strength of this same "
                 "ClinVar record, so it bypassed the frequency and pathogenicity "
                 "filters. Its survival to the shortlist is therefore not evidence "
                 "independent of ClinVar."]
                if dig(ve, "whiteListed") is True else []
            ),
        ), supporting=not conflicted)
    else:
        cand.missing_evidence.append("no ClinVar record in 2406 snapshot")

    # --- population frequency ----------------------------------------------
    state, af, raw = frequency_state(ve)
    if state == "absent":
        attach(log.add(
            candidate_id=cid,
            category="gnomad",
            source="Exomiser frequency sources (gnomAD, ExAC, ESP, 1000G)",
            query=f"{cand.chrom}:{cand.pos} {cand.ref}>{cand.alt}",
            raw_field="frequencyData",
            raw_value=raw or '{"empty": true}',
            tool_or_data_version=PROVENANCE,
            interpretation=(
                "Not observed in any bundled population database. Consistent with "
                "PM2 (absent from controls) for a rare dominant disorder."
            ),
            limitations=[
                _SNAPSHOT_CAVEAT,
                "Absence in the snapshot is not absence in gnomAD v4 - confirm live.",
                "Absent is not the same as allele frequency 0.0; no AF is recorded.",
            ],
        ))
        cand.missing_evidence.append("gnomAD v4 allele frequency not yet queried live")
    elif state == "present":
        cand.gnomad_af = af
        common = af is not None and af > COMMON_AF_PERCENT
        attach(log.add(
            candidate_id=cid,
            category="gnomad",
            source="Exomiser frequency sources",
            query=f"{cand.chrom}:{cand.pos} {cand.ref}>{cand.alt}",
            raw_field="frequencyData.frequencies",
            raw_value=raw,
            tool_or_data_version=PROVENANCE,
            interpretation=(
                f"Observed in population controls at a maximum frequency of {af}%. "
                + (f"Above Exomiser's {COMMON_AF_PERCENT}% cutoff - too common for a "
                   "rare dominant disorder."
                   if common else
                   "Rare, but PRESENT - PM2 (absent from controls) must NOT be "
                   "claimed for this variant.")
            ),
            limitations=[
                _SNAPSHOT_CAVEAT,
                "Rarity is not pathogenicity; most rare variants are benign.",
            ],
        ), supporting=not common)
    else:
        cand.missing_evidence.append("population frequency never queried")

    # --- effect predictors --------------------------------------------------
    scores = predictor_scores(ve)
    cand.effect_scores.update(scores)
    for source, value in sorted(scores.items()):
        attach(log.add(
            candidate_id=cid,
            category="effect",
            source=source,
            query=f"{cand.chrom}:{cand.pos} {cand.ref}>{cand.alt}",
            transcript=str(accession) if accession else None,
            raw_field="score",
            raw_value=f"{value}",
            tool_or_data_version=PROVENANCE,
            interpretation=f"{source} score {value} (higher = more likely damaging).",
            limitations=[
                "In silico prediction only - no functional experiment.",
                "Predictors share training data; agreement is not independent replication.",
            ],
        ))
    if not scores:
        cand.missing_evidence.append("no missense/splice predictor scores available")

    # --- phenotype match ----------------------------------------------------
    pheno, pheno_path = gene_phenotype_score(gene_rec)
    if pheno is not MISSING and isinstance(pheno, (int, float)):
        # Real values are 0.500024423648938, not a clean 0.5 - an exact-equality
        # test silently promotes the placeholder to genuine supporting evidence.
        default_like = abs(float(pheno) - NO_GENE_DISEASE_LINK) < 1e-3
        cand.phenotype_match = {
            "score": float(pheno),
            "source_field": ".".join(str(p) for p in (pheno_path or ())),
            "is_exomiser_default": default_like,
        }
        attach(log.add(
            candidate_id=cid,
            category="phenotype",
            source="Exomiser hiPHIVE / HPO",
            record_or_accession=cand.gene or "",
            query="patient HPO: " + ", ".join(sorted(PATIENT_HPO)),
            raw_field=".".join(str(p) for p in (pheno_path or ())),
            raw_value=f"{pheno}",
            tool_or_data_version=PROVENANCE,
            interpretation=(
                f"Phenotype-match score {pheno} for {cand.gene}. "
                + ("This equals Exomiser's default for 'no known gene-disease "
                   "relationship' - treat as absence of data, not as a measurement."
                   if default_like else
                   "Derived from the gene's known disease HPO annotations.")
            ),
            limitations=(
                ["0.5 is a placeholder, not a match."] if default_like else
                ["Scores the gene, not this specific variant."]
            ),
        ), supporting=not default_like)
    else:
        cand.missing_evidence.append("phenotype score not found in record")

    # --- ACMG assignment ----------------------------------------------------
    acmg = dig(gene_score_rec, "acmgAssignments", 0)
    if isinstance(acmg, dict):
        classification = acmg.get("acmgClassification")
        # The criteria live one level deeper: acmgEvidence.evidence = {"PM2": ...}
        criteria, _ = first_present(
            acmg, [("acmgEvidence", "evidence"), ("acmgEvidence",)]
        )
        criteria = criteria if criteria is not MISSING else {}
        disease = dig(acmg, "disease", "diseaseId")
        moi = acmg.get("modeOfInheritance")

        # --- disease HPO overlap: named terms, not just a score ---------------
        disease_hpo = dig(acmg, "disease", "phenotypeIds")
        if isinstance(disease_hpo, list) and disease_hpo:
            matched = [h for h in PATIENT_HPO if h in set(disease_hpo)]
            unmatched = [h for h in PATIENT_HPO if h not in set(disease_hpo)]
            disease_name = dig(acmg, "disease", "diseaseName")
            attach(log.add(
                candidate_id=cid,
                category="phenotype",
                source="OMIM disease annotation (via Exomiser 2406)",
                record_or_accession=str(disease) if disease is not MISSING else "",
                query="patient HPO: " + ", ".join(sorted(PATIENT_HPO)),
                raw_field="disease.phenotypeIds",
                raw_value=f"{len(disease_hpo)} terms; matched {len(matched)}/6",
                tool_or_data_version=PROVENANCE,
                url=(f"https://omim.org/entry/{str(disease).split(':')[-1]}"
                     if isinstance(disease, str) and disease.startswith("OMIM:") else ""),
                interpretation=(
                    f"{len(matched)}/6 patient HPO terms appear in the annotation for "
                    f"{disease} ({disease_name}): "
                    + ", ".join(f"{h} {PATIENT_HPO[h]}" for h in matched)
                    + (f". Not annotated: "
                       + ", ".join(f"{h} {PATIENT_HPO[h]}" for h in unmatched)
                       if unmatched else "")
                ),
                limitations=[
                    "Term-ID overlap only - no ontology ancestor matching, so a "
                    "parent/child term counts as a miss.",
                    "Disease-level annotation; it scores the gene, not this variant.",
                    _SNAPSHOT_CAVEAT,
                ],
            ), supporting=bool(matched))
        attach(log.add(
            candidate_id=cid,
            category="clingen",
            source="Exomiser ACMG/AMP auto-assignment",
            record_or_accession=str(disease) if disease is not MISSING else "",
            query=f"{cand.gene} / {moi}",
            raw_field="acmgClassification | acmgEvidence.evidence",
            raw_value=f"{classification} | {json.dumps(criteria, sort_keys=True)[:300]}",
            tool_or_data_version=PROVENANCE,
            interpretation=(
                f"Automated ACMG classification: {classification} under {moi}."
            ),
            limitations=[
                "Automated, not expert-panel curated - not a ClinGen VCEP call.",
                "PP5 ('reputable source reports pathogenic') is derived from the same "
                "ClinVar record cited separately above; ClinGen deprecated PP5 for this "
                "circularity. Do not count ClinVar twice.",
                _SNAPSHOT_CAVEAT,
            ],
        ))
    else:
        cand.missing_evidence.append("no ACMG assignment in record")


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #


def build(
    records: list[dict],
    top_genes: int = 10,
    max_variants_per_gene: int = 3,
) -> tuple[list[Candidate], list[Evidence], list[DropRecord]]:
    assert_no_leak(records)

    ranked = sorted(
        records,
        key=lambda r: (dig(r, "combinedScore") if isinstance(dig(r, "combinedScore"), (int, float)) else -1.0),
        reverse=True,
    )

    log = EvidenceLog()
    candidates: list[Candidate] = []
    drops: list[DropRecord] = []

    for rank, rec in enumerate(ranked, 1):
        symbol = gene_symbol(rec)
        variants = dig(rec, "variantEvaluations")
        variants = variants if isinstance(variants, list) else []
        gene_score_rec = dig(rec, "geneScores", 0)
        gene_score_rec = gene_score_rec if isinstance(gene_score_rec, dict) else {}

        # Genes below the cutoff are dropped WITH A REASON, not silently truncated.
        if rank > top_genes:
            score = dig(rec, "combinedScore")
            for ve in variants:
                cid = _cid_for(ve)
                if cid:
                    drops.append(DropRecord(
                        candidate_id=cid,
                        stage="shortlist",
                        reason=(
                            f"gene {symbol} ranked {rank} by Exomiser combinedScore "
                            f"({score}), below the top-{top_genes} shortlist cutoff"
                        ),
                    ))
            continue

        kept_for_gene = 0
        for ve in variants:
            cid = _cid_for(ve)
            if not cid:
                continue

            status = dig(ve, "filterStatus")
            if isinstance(status, str) and status.upper() == "FAILED":
                failed = dig(ve, "failedFilterTypes")
                reason = ", ".join(failed) if isinstance(failed, list) and failed else "unspecified"
                drops.append(DropRecord(
                    candidate_id=cid,
                    stage="exomiser_filter",
                    reason=f"failed Exomiser filter(s): {reason}",
                ))
                continue

            if kept_for_gene >= max_variants_per_gene:
                drops.append(DropRecord(
                    candidate_id=cid,
                    stage="shortlist",
                    reason=(
                        f"more than {max_variants_per_gene} passing variants in {symbol}; "
                        "kept the highest variantScore only"
                    ),
                ))
                continue

            candidates.append(_make_candidate(cid, ve, rec, symbol, rank, log, gene_score_rec))
            kept_for_gene += 1

    return candidates, log.rows, drops


def _cid_for(ve: dict) -> Optional[str]:
    chrom = dig(ve, "contigName")
    if chrom is MISSING:
        chrom = dig(ve, "chromosomeName")
    pos, _ = first_present(ve, [("start",), ("pos",), ("position",)])
    ref = dig(ve, "ref")
    alt = dig(ve, "alt")
    if MISSING in (chrom, pos, ref, alt):
        return None
    return candidate_id(norm_chrom(chrom), int(pos), str(ref), str(alt))


def _make_candidate(
    cid: str, ve: dict, rec: dict, symbol: str, rank: int,
    log: EvidenceLog, gene_score_rec: dict,
) -> Candidate:
    chrom, pos, ref, alt = cid.split("-", 3)
    tx = best_transcript(ve)
    qual, _ = first_present(ve, [("phredScore",), ("qual",)])
    eff = variant_effect(ve)

    cand = Candidate(
        candidate_id=cid,
        chrom=chrom,
        pos=int(pos),
        ref=ref,
        alt=alt,
        rsid=(lambda r: str(r) if r not in (MISSING, None, "", ".") else None)(
            first_present(ve, [("frequencyData", "rsId"), ("id",), ("rsId",)])[0]
        ),
        zygosity=zygosity_of(ve),
        qual=float(qual) if isinstance(qual, (int, float)) else None,
        depth=None,  # Exomiser output carries no DP - stays None, never 0
        var_type=var_type_of(ref, alt),
        tier="",
        filter_trail=[f"exomiser:PASS", f"exomiser:gene_rank={rank}"],
        gene=symbol or None,
        transcript=str(tx.get("accession")) if tx.get("accession") else None,
        consequence=str(eff) if eff is not MISSING else None,
        hgvs_c=tx.get("hgvsCdna") or None,
        hgvs_p=tx.get("hgvsProtein") or None,
    )
    cand.missing_evidence.append("read depth not available from Exomiser output")
    cand.missing_evidence.append(
        "no transcript accession in Exomiser output - MANE Select must come from VEP"
    )
    if dig(ve, "whiteListed") is True:
        cand.filter_trail.append("exomiser:whitelisted(ClinVar P/LP - filters bypassed)")
    harvest_evidence(log, cand, rec, ve, gene_score_rec)
    return cand


# --------------------------------------------------------------------------- #
# Inspect mode - run this before trusting anything above
# --------------------------------------------------------------------------- #


def inspect(records: list[dict], limit: int = 3) -> None:
    print(f"records: {len(records)}")
    print(f"top-level keys: {sorted(records[0].keys()) if records else '-'}\n")
    for rec in records[:limit]:
        symbol = gene_symbol(rec)
        combined = dig(rec, "combinedScore")
        pheno, path = gene_phenotype_score(rec)
        variants = dig(rec, "variantEvaluations")
        variants = variants if isinstance(variants, list) else []
        print(f"--- {symbol}  combinedScore={combined}")
        print(f"    phenotype score {pheno} from key path "
              f"{'.'.join(str(p) for p in path) if path else 'NOT FOUND'}")
        print(f"    variantEvaluations: {len(variants)}")
        if variants:
            ve = variants[0]
            print(f"    variant keys: {sorted(ve.keys())}")
            print(f"    id: {_cid_for(ve)}  effect: {variant_effect(ve)}")
            print(f"    frequency: {frequency_state(ve)[0]}")
            print(f"    predictors: {predictor_scores(ve)}")
            cv = clinvar_block(ve)
            print(f"    clinvar keys: {sorted(cv.keys()) if cv else 'none'}")
        gs = dig(rec, "geneScores", 0)
        if isinstance(gs, dict):
            print(f"    geneScores[0] keys: {sorted(gs.keys())}")
        print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Exomiser JSONL -> candidates + evidence")
    ap.add_argument("jsonl", help="path to Pfeiffer-exomiser.jsonl")
    ap.add_argument("--outdir", default="outputs/blockB")
    ap.add_argument("--top-genes", type=int, default=10)
    ap.add_argument("--max-variants-per-gene", type=int, default=3)
    ap.add_argument("--inspect", action="store_true",
                    help="print the real key structure and exit")
    args = ap.parse_args(argv)

    records = load_records(args.jsonl)

    if args.inspect:
        inspect(records)
        return 0

    candidates, evidence, drops = build(
        records, args.top_genes, args.max_variants_per_gene
    )

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "candidates.json"), "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in candidates], fh, indent=2)
    for name, rows in (("evidence.jsonl", evidence), ("drops.jsonl", drops)):
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict()) + "\n")

    print(f"candidates : {len(candidates)}")
    print(f"evidence   : {len(evidence)}")
    print(f"drops      : {len(drops)}")
    print(f"written to : {args.outdir}/")
    for c in candidates[:10]:
        print(f"  {c.gene:<10} {c.candidate_id:<24} {c.consequence or '-':<22}"
              f" clinvar={c.clinvar_classification or '-'}"
              f" missing={len(c.missing_evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())