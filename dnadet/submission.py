"""Block F - assemble submission.json and validate it against the template.

Everything in the output is derived from the evidence log. Nothing is authored
here: `reason_for_rank` comes from the same `assess()` that produced the
ranking, the follow-up answers come from the same `qa` module the user talks to,
and every evidence ID cited by a candidate is checked to exist before the file
is written.

Two fields need honest handling
-------------------------------
`confidence` is a float in the template, which invites reading it as a
probability. It is not one. Nothing here is calibrated against outcomes - there
is one patient and no ground truth. It is an ordinal transform of evidence
strength, and the file says so in `limitations` rather than letting the number
imply more than it can support.

`hgvs_g` is emitted only for substitutions, where the notation is unambiguous.
Indel HGVS requires left-alignment against the reference sequence, which this
system does not do; a plausible-looking wrong HGVS is worse than an empty
string.

Usage
-----
    python3 -m dnadet.submission --team "Team 9"
    python3 -m dnadet.submission --team "Team 9" --qa-policy llm --backend groq
    python3 -m dnadet.submission --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Optional

from .agent import assess, load_rows, now_iso
from .contract import ASSEMBLY, HPO_TERMS, SAMPLE
from .qa import answer as qa_answer

AGENT_NAME = "DNA Detective"
AGENT_VERSION = "1.0.0"

FOLLOW_UPS = [
    "Why is candidate 1 more suspicious than candidate 2?",
    "What evidence argues against your top candidate?",
]

HPO_RE = re.compile(r"HP:\d{7}")
PATIENT_HPO = dict(HPO_TERMS)

# Ordinal, not probabilistic. See the module docstring and `limitations`.
CONFIDENCE_SCALE = {0: 0.05, 1: 0.20, 2: 0.35, 3: 0.50, 4: 0.65, 5: 0.75,
                    6: 0.85, 7: 0.90}


def confidence_for(strength: int) -> float:
    return CONFIDENCE_SCALE.get(min(strength, 7), 0.90)


def hgvs_g(cand: dict) -> str:
    """Genomic HGVS - substitutions only. See module docstring."""
    ref, alt = cand.get("ref", ""), cand.get("alt", "")
    if len(ref) == 1 and len(alt) == 1 and ref != alt:
        return f"{cand['chrom']}:g.{cand['pos']}{ref}>{alt}"
    return ""


def disease_and_inheritance(cid: str, rows: list[dict]) -> tuple[str, str]:
    """Pull the candidate disease and mode of inheritance from ACMG evidence."""
    disease, moi = "", ""
    for r in rows:
        if r.get("candidate_id") != cid:
            continue
        if r.get("category") == "clingen" and "ACMG" in (r.get("source") or ""):
            acc = r.get("record_or_accession") or ""
            if acc.startswith("OMIM:") and not disease:
                disease = acc
            q = r.get("query") or ""
            if "/" in q and not moi:
                moi = q.split("/", 1)[1].strip()
        if not disease and r.get("category") == "phenotype" and \
                (r.get("record_or_accession") or "").startswith("OMIM:"):
            disease = r["record_or_accession"]
            m = re.search(r"\(([^)]+)\)", r.get("interpretation") or "")
            if m and not disease.endswith(")"):
                disease = f"{disease} ({m.group(1)})"
    return disease, moi


def matched_terms(cid: str, rows: list[dict]) -> list[str]:
    """Which of the patient's six terms appear in the gene's disease annotation."""
    found: set[str] = set()
    for r in rows:
        if r.get("candidate_id") != cid or r.get("category") != "phenotype":
            continue
        text = (r.get("interpretation") or "")
        if "patient HPO terms appear" not in text:
            continue
        head = text.split("Not annotated")[0]
        found.update(t for t in HPO_RE.findall(head) if t in PATIENT_HPO)
    return sorted(found)


def phenotype_explanation(cid: str, rows: list[dict], matched: list[str]) -> str:
    for r in rows:
        if r.get("candidate_id") == cid and r.get("category") == "phenotype" \
                and "patient HPO terms appear" in (r.get("interpretation") or ""):
            return r["interpretation"]
    if matched:
        return f"{len(matched)}/6 patient terms matched."
    return ("No disease-term overlap with the patient's phenotype; any phenotype "
            "score derives from protein-interaction channels, not term matching.")


def tools_used(rows: list[dict]) -> list[str]:
    seen: dict[str, str] = {}
    for r in rows:
        src = (r.get("source") or "").strip()
        ver = (r.get("tool_or_data_version") or "").strip()
        if src and src not in seen:
            seen[src] = ver
    return [f"{s} ({v})" if v else s for s, v in sorted(seen.items())]


def build(candidates: list[dict], rows: list[dict], team: str, top_n: int,
          qa_policy: str, backend: str, model: str,
          started: str) -> tuple[dict, list[str]]:
    ranked = sorted([assess(c, rows) for c in candidates], key=lambda a: a.sort_key)
    by_id = {c["candidate_id"]: c for c in candidates}
    problems: list[str] = []
    valid_ids = {r.get("evidence_id") for r in rows if r.get("evidence_id")}

    top: list[dict] = []
    for rank, a in enumerate(ranked[:top_n], 1):
        c = by_id[a.candidate_id]
        support = sorted({e for v in a.verdicts if v.stance == "supports"
                          for e in v.evidence_ids})
        conflict = sorted({e for v in a.verdicts if v.stance == "conflicts"
                           for e in v.evidence_ids})
        for eid in support + conflict:
            if eid not in valid_ids:
                problems.append(f"{a.candidate_id} cites {eid}, absent from the log")

        matched = matched_terms(a.candidate_id, rows)
        disease, moi = disease_and_inheritance(a.candidate_id, rows)
        reason = a.reason
        if a.circularity:
            reason += " Circularity noted: " + " ".join(a.circularity)

        top.append({
            "rank": rank,
            "candidate_id": a.candidate_id,
            "normalized_variant": {
                "assembly": ASSEMBLY,
                "chrom": c["chrom"], "pos": c["pos"],
                "ref": c["ref"], "alt": c["alt"],
                "hgvs_g": hgvs_g(c),
                "hgvs_c": c.get("hgvs_c") or "",
                "hgvs_p": c.get("hgvs_p") or "",
            },
            "gene": c.get("gene") or "",
            "transcript": c.get("transcript") or "",
            "consequence": c.get("consequence") or "",
            "zygosity": c.get("zygosity") or "",
            "candidate_disease": disease,
            "inheritance": moi,
            "phenotype_match": {
                "score": (c.get("phenotype_match") or {}).get("score"),
                "matched_hpo_terms": matched,
                "explanation": phenotype_explanation(a.candidate_id, rows, matched),
            },
            "supporting_evidence_ids": support,
            "conflicting_evidence_ids": conflict,
            "missing_evidence": a.gaps,
            "reason_for_rank": reason,
            "confidence": confidence_for(a.strength),
        })

    follow_ups = [{"user": q,
                   "agent": qa_answer(q, ranked, rows, qa_policy, backend, model)}
                  for q in FOLLOW_UPS]

    limitations = [
        "Educational exercise. Not a validated clinical system; not for patient care.",
        "`confidence` is an ORDINAL transform of evidence strength, not a "
        "calibrated probability. There is one patient and no ground truth, so no "
        "calibration is possible.",
        "Evidence weights are ordinal judgements: expert-reviewed clinical "
        "evidence and phenotype term overlap outweigh in-silico prediction.",
        "`hgvs_g` is emitted for substitutions only. Indel HGVS requires "
        "left-alignment against the reference, which this system does not perform.",
        "Exomiser's combinedScore is never cited as evidence. It is reproducible "
        "only with the ACMG posterior as a third input, so its derivation cannot "
        "be shown from the two published components.",
        "One VCF record carries GENE=/INHERITANCE=/MIM= INFO fields and a phased "
        "genotype, leaking the intended answer. These fields are blocked by an "
        "assertion in the parser and were never read; the leak is disclosed here "
        "rather than used.",
        "Per-variant drop reasons exist only for the 282 variants that survived "
        "Exomiser's cascade. For the 37,427 removed earlier, reasons are "
        "available at filter-stage granularity only, and the published stage "
        "counts account for 37,093 - a discrepancy of 334 records that enter no "
        "stage, disclosed rather than reconciled.",
        "Follow-up answers phrased by a language model are constrained to the "
        "retrieved evidence and their citation IDs are mechanically verified "
        "against the log. Claim-to-source fidelity within a verified citation is "
        "NOT machine-checked.",
        "Population frequencies come from the Exomiser 2406 snapshot unless an "
        "evidence row states a live retrieval. Absence in that snapshot is not "
        "absence in gnomAD v4.",
    ]

    return {
        "team": team,
        "case_id": SAMPLE,
        "input": {
            "vcf": "data/Pfeiffer.vcf",
            "phenopacket": "data/pfeiffer-phenopacket.yml",
            "assembly": ASSEMBLY,
            "sample": SAMPLE,
            "hpo_terms": [h for h, _ in HPO_TERMS],
        },
        "method": {
            "agent_name": AGENT_NAME,
            "agent_version": AGENT_VERSION,
            "tools": tools_used(rows),
            "run_started_at": started,
            "run_finished_at": now_iso(),
        },
        "top_candidates": top,
        "evidence_log": rows,
        "follow_up_examples": follow_ups,
        "limitations": limitations,
    }, problems


# --------------------------------------------------------------------------- #
# Validation against the starter template
# --------------------------------------------------------------------------- #


def shape(obj: Any, path: str = "") -> set[str]:
    """Every key path in a nested structure, so shapes can be compared."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            keys.add(p)
            keys |= shape(v, p)
    elif isinstance(obj, list) and obj:
        keys |= shape(obj[0], f"{path}[]")
    return keys


def validate(doc: dict, template_path: str) -> list[str]:
    issues: list[str] = []
    if not os.path.exists(template_path):
        return [f"template not found at {template_path} - shape unverified"]
    with open(template_path, "r", encoding="utf-8") as fh:
        template = json.load(fh)

    missing = shape(template) - shape(doc)
    for k in sorted(missing):
        issues.append(f"missing key required by template: {k}")

    ids = [r.get("evidence_id") for r in doc.get("evidence_log", [])]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        issues.append(f"duplicate evidence IDs: {sorted(dupes)}")
    valid = set(ids)
    for c in doc.get("top_candidates", []):
        for eid in c.get("supporting_evidence_ids", []) + \
                c.get("conflicting_evidence_ids", []):
            if eid not in valid:
                issues.append(f"candidate {c['candidate_id']} cites unknown {eid}")
    for c in doc.get("top_candidates", []):
        if not c.get("reason_for_rank"):
            issues.append(f"candidate {c['candidate_id']} has no reason_for_rank")
    for f in doc.get("follow_up_examples", []):
        if not f.get("agent"):
            issues.append(f"follow-up unanswered: {f.get('user')}")
    return issues


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Block F - build submission.json")
    ap.add_argument("--team", default="TEAM_NAME")
    ap.add_argument("--candidates", default="outputs/blockB/candidates.json")
    ap.add_argument("--evidence", nargs="*",
                    default=["outputs/blockB/evidence.jsonl",
                             "outputs/blockC/evidence.jsonl",
                             "outputs/blockE/agent_evidence.jsonl"])
    ap.add_argument("--template", default="starter/submission_template.json")
    ap.add_argument("--out", default="outputs/submission.json")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--qa-policy", choices=["template", "llm"], default="template")
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--model", default="")
    ap.add_argument("--validate-only", action="store_true",
                    help="re-validate an existing submission.json")
    args = ap.parse_args(argv)

    if args.validate_only:
        with open(args.out, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        issues = validate(doc, args.template)
        print(f"{len(issues)} issue(s)")
        for i in issues:
            print(f"  - {i}")
        return 1 if issues else 0

    started = now_iso()
    with open(args.candidates, "r", encoding="utf-8") as fh:
        candidates = json.load(fh)
    rows = load_rows(args.evidence)
    print(f"{len(candidates)} candidates, {len(rows)} evidence rows")

    doc, problems = build(candidates, rows, args.team, args.top_n,
                          args.qa_policy, args.backend, args.model, started)
    for p in problems:
        print(f"  ! {p}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    issues = validate(doc, args.template)
    print(f"\ntop candidates : {len(doc['top_candidates'])}")
    print(f"evidence log   : {len(doc['evidence_log'])}")
    print(f"tools recorded : {len(doc['method']['tools'])}")
    print(f"validation     : {'PASS' if not issues else str(len(issues)) + ' issue(s)'}")
    for i in issues:
        print(f"  - {i}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())