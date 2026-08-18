#!/usr/bin/env python3
"""Entry point for the DNA Detective MVP skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.case_loader import load_case
from src.pipeline.prioritization import generate_candidate_shortlist
from src.pipeline.reporting import build_report, write_report
from src.tools import discover_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path, default=Path("data/Pfeiffer.vcf"))
    parser.add_argument("--phenopacket", type=Path, default=Path("data/pfeiffer-phenopacket.yml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dna_detective_report.json"))
    parser.add_argument("--team", default="TEAM_NAME")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write an intentionally empty report; does not analyze variants.")
    args = parser.parse_args()
    case = load_case(args.vcf, args.phenopacket)
    registry = discover_tools()
    if args.dry_run:
        report = build_report(case=case, candidates=[], evidence=[], tools=registry.names(), team=args.team, limitations=["Dry run only: candidate filtering, external evidence, and agent reasoning are intentionally not implemented."])
    else:
        candidates = generate_candidate_shortlist(case, limit=args.limit)
        # TODO: select tools, investigate candidates, reason over evidence, then build the report.
        report = build_report(case, candidates, evidence=[], tools=registry.names(), team=args.team, limitations=["External evidence and agent reasoning have not yet been wired in."])
    write_report(report, args.output)
    print(f"Wrote report skeleton to {args.output}")


if __name__ == "__main__":
    main()

