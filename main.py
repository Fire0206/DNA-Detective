#!/usr/bin/env python3
"""Entry point for DNA Detective: VCF + phenotypes -> ranked candidates with traceable evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agent.investigator import follow_up_examples, investigate_candidates, reason_over_evidence
from src.pipeline.case_loader import load_case
from src.pipeline.prioritization import generate_candidate_shortlist, shortlist_drops, shortlist_evidence
from src.pipeline.reporting import build_report, write_report
from src.tools import discover_tools

LIMITATIONS = [
    "Educational exercise. Not a validated clinical system; not for patient care.",
    "`confidence` is an ORDINAL transform of evidence strength, not a calibrated probability. One patient, no ground truth, so no calibration is possible.",
    "Evidence weights are ordinal judgements: expert-reviewed clinical evidence and phenotype-term overlap outweigh in-silico prediction.",
    "Exomiser's combinedScore is never cited as evidence. It is reproducible only with the ACMG posterior as a third input, so its derivation cannot be shown from the two published components.",
    "One VCF record carries GENE=/INHERITANCE=/MIM= INFO fields and a phased genotype, leaking the intended answer. These fields are blocked by an assertion in the parser and were never read; the leak is disclosed here rather than used.",
    "Per-variant drop reasons exist only for variants that survived Exomiser's cascade. For those removed earlier, reasons are available at filter-stage granularity only, and the published stage counts account for 37,093 of 37,427 removals - a 334-record discrepancy disclosed rather than reconciled.",
    "Population frequencies come from the Exomiser 2406 snapshot unless an evidence row states a live retrieval. Absence in that snapshot is not absence in gnomAD v4.",
    "Consequence annotation is available only for variants carrying a ClinVar record; this Exomiser output contains no transcript annotations. Candidates without one cannot have their molecular effect confirmed and are marked accordingly.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path, default=Path("data/Pfeiffer.vcf"))
    parser.add_argument("--phenopacket", type=Path, default=Path("data/pfeiffer-phenopacket.yml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dna_detective_report.json"))
    parser.add_argument("--team", default="TEAM_NAME")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tools", default="", help="Comma-separated tool names; default is every discovered tool.")
    parser.add_argument("--no-tools", action="store_true", help="Skip live lookups; rank on offline evidence only.")
    parser.add_argument("--qa-policy", choices=["template", "llm"], default="template", help="How follow-up answers are phrased; retrieval is deterministic either way.")
    parser.add_argument("--qa-backend", default="groq")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write an intentionally empty report; does not analyze variants.")
    args = parser.parse_args()

    case = load_case(args.vcf, args.phenopacket)
    registry = discover_tools()

    if args.dry_run:
        report = build_report(case=case, candidates=[], evidence=[], tools=registry.names(), team=args.team, limitations=["Dry run only: candidate filtering, external evidence, and agent reasoning were not executed."])
        write_report(report, args.output)
        print(f"Wrote dry-run report to {args.output}")
        return

    candidates = generate_candidate_shortlist(case, limit=args.limit)
    evidence = list(shortlist_evidence())
    print(f"shortlist: {len(candidates)} candidates, {len(evidence)} offline evidence rows, {len(shortlist_drops())} drops with reasons")

    selected = [] if args.no_tools else (args.tools.split(",") if args.tools else registry.names())
    if selected:
        print(f"tools: {', '.join(selected)}")
        evidence.extend(investigate_candidates(candidates, case, selected, registry))
        print(f"evidence after live lookups: {len(evidence)} rows")

    ranked = reason_over_evidence(candidates, evidence)
    report = build_report(case=case, candidates=ranked, evidence=evidence, tools=registry.names(), team=args.team, limitations=LIMITATIONS)
    report.follow_up_examples = follow_up_examples(ranked, evidence, policy=args.qa_policy, backend=args.qa_backend)
    write_report(report, args.output)

    print("\nranking:")
    for candidate in ranked[:5]:
        print(f"  {candidate.rank}. {candidate.gene or '?':<8} {candidate.candidate_id:<26} confidence={candidate.confidence}")
    print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
