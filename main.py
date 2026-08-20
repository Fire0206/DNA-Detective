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


# --------------------------------------------------------------------------- #
# Agent-mode: ClinVar tool adapter
# --------------------------------------------------------------------------- #

def _make_clinvar_agent_tool(cache_path: str):
    """Wrap clinical.process as a per-candidate tool for the agent loop.

    Side-effect: updates the candidate dict with any classification found so
    that assess() sees it on the next OBSERVE step. The dict is passed by
    reference through cands[candidate_id] in the loop, so mutations persist.
    """
    def _tool(cand: dict):
        from dnadet.tools.clinical import process as clinvar_process, normalise_label
        rows, _ = clinvar_process([cand], cache_path, replay=False, limit=1)
        for row in rows:
            d = row.to_dict()
            if d.get("category") != "clinvar" or d.get("source") != "ClinVar (live)":
                continue
            raw = (d.get("raw_value") or "")
            parts = [p.strip() for p in raw.split(" | ")]
            if not parts or not parts[0]:
                continue
            # Only set if candidate has no snapshot classification
            if not cand.get("clinvar_classification"):
                cand["clinvar_classification"] = normalise_label(parts[0])
            if len(parts) > 1 and cand.get("clinvar_review_stars") is None:
                status = parts[1].lower()
                if "expert" in status:
                    cand["clinvar_review_stars"] = 3
                elif "multiple" in status:
                    cand["clinvar_review_stars"] = 2
                elif "single" in status or "submitter" in status:
                    cand["clinvar_review_stars"] = 1
                else:
                    cand["clinvar_review_stars"] = 0
        return rows
    return _tool


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
    # --- agent mode ---
    parser.add_argument("--agent", action="store_true", help="Run the ReAct agent loop (agent decides which tools to call).")
    parser.add_argument("--agent-policy", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--agent-steps", type=int, default=10)
    parser.add_argument("--agent-pace", type=float, default=0.0, help="Seconds between agent steps (try 8 on Groq free tier).")
    parser.add_argument("--transcript", type=Path, default=Path("outputs/agent_transcript.md"),
                        help="Where to write the agent transcript. Defaults under outputs/ because it is "
                             "a generated artifact; point it at docs/transcripts/ to keep a run as a deliverable.")
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

    # ------------------------------------------------------------------ #
    # PATH A: agent-driven pipeline                                       #
    # ------------------------------------------------------------------ #
    if args.agent:
        from dnadet import agent as agent_mod
        from src import bridge

        # Register live tools the agent can decide to call
        agent_mod.TOOLS["clinvar"] = _make_clinvar_agent_tool("outputs/agent_cache")

        from dnadet.tools.vep import annotate_candidate as vep_annotate
        agent_mod.TOOLS["vep"] = lambda cand: vep_annotate(cand, cache_dir="outputs/agent_cache/vep")

        # Convert to engine format
        cand_dicts = [bridge.to_engine_dict(c) for c in candidates]
        ev_dicts = bridge.evidence_as_dicts(evidence)

        policy = (agent_mod.DeterministicPolicy() if args.agent_policy == "deterministic"
                  else agent_mod.LLMPolicy(args.qa_backend))
        print(f"agent policy: {policy.name}")
        transcript: list[str] = []
        final, new_rows = agent_mod.run(cand_dicts, ev_dicts, policy,
                                        args.agent_steps, transcript, args.agent_pace)

        # Write transcript. This is a generated artifact, so it defaults under
        # outputs/ (gitignored) rather than into a tracked docs/ path - otherwise
        # every agent run leaves the tree dirty with a re-timestamped file.
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        agent_mod.write_transcript(str(args.transcript), transcript, final)
        print(f"agent transcript -> {args.transcript}")

        # Convert assessments back to model candidates
        cand_map = {c["candidate_id"]: c for c in cand_dicts}
        max_str = max((a.strength for a in final), default=1) or 1
        ranked = []
        for i, a in enumerate(final, 1):
            rec = dict(cand_map[a.candidate_id])
            rec["reason_for_rank"] = a.reason
            rec["confidence"] = round(a.strength / max_str, 2) if a.strength > 0 else 0.0
            ranked.append(bridge.to_model_candidate(rec, rank=i))

        # Merge agent-produced evidence into the offline set
        agent_ev = bridge.to_model_evidence(new_rows, prefix="A")
        evidence.extend(agent_ev)
        print(f"evidence after agent: {len(evidence)} rows ({len(agent_ev)} from agent tools)")

        # Q&A will use the same live tools to fill gaps and investigate ad-hoc variants
        qa_kwargs: dict = {"cands": cand_map, "tools": agent_mod.TOOLS}

    # ------------------------------------------------------------------ #
    # PATH B: linear pipeline (original behaviour)                        #
    # ------------------------------------------------------------------ #
    else:
        selected = [] if args.no_tools else (args.tools.split(",") if args.tools else registry.names())
        if selected:
            print(f"tools: {', '.join(selected)}")
            evidence.extend(investigate_candidates(candidates, case, selected, registry))
            print(f"evidence after live lookups: {len(evidence)} rows")

        ranked = reason_over_evidence(candidates, evidence)
        qa_kwargs = {}

    # ------------------------------------------------------------------ #
    # Common: build and write the report                                  #
    # ------------------------------------------------------------------ #
    report = build_report(case=case, candidates=ranked, evidence=evidence, tools=registry.names(), team=args.team, limitations=LIMITATIONS)
    report.follow_up_examples = follow_up_examples(ranked, evidence, policy=args.qa_policy, backend=args.qa_backend, **qa_kwargs)
    write_report(report, args.output)

    print("\nranking:")
    for candidate in ranked[:5]:
        print(f"  {candidate.rank}. {candidate.gene or '?':<8} {candidate.candidate_id:<26} confidence={candidate.confidence}")
    print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
