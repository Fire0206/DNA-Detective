"""Block G - follow-up questions, answered from the evidence log.

README item 10: answer a question such as "Why is candidate 1 stronger than
candidate 2?" This module does that, and does it without letting a language
model invent anything.

The division of labour
----------------------
RETRIEVAL and COMPARISON are deterministic. Which candidates a question is
about, what evidence each holds, which families support or conflict, where the
gaps are - all computed from the same `assess()` used for ranking. Identical
every run, auditable, no model involved.

PHRASING is the model's job, and only phrasing. It receives the retrieved rows
and nothing else, and is instructed to add no facts.

CITATIONS ARE THEN VERIFIED. Every evidence ID the model emits is checked
against the log before the answer is shown. An ID that does not exist is
stripped and reported. A fabricated citation in a genetics report is worse than
no answer at all, because it looks exactly like a real one - the whole point of
Block C's locus and allele checks was to stop precisely this, and it would be
incoherent to let the answer layer reintroduce it.

Without a model, the templated answer says the same thing less fluently, from
the same comparison, with the same citations.

INVESTIGATION MODE (new): when ``cands`` and ``tools`` are provided, Q&A
becomes an agent. It can fill evidence gaps before answering, and investigate
ad-hoc variants that were never in the Exomiser shortlist. This directly
answers the prof's feedback: "if a user asks about a specific variant,
investigate it" and "reconsider candidates that Exomiser ranks lower."

Usage
-----
    python3 -m dnadet.qa --ask "why is candidate 1 stronger than candidate 2"
    python3 -m dnadet.qa --interactive
    python3 -m dnadet.qa --ask "what is wrong with COL4A1" --policy llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from .agent import BACKENDS, Assessment, assess, load_rows, now_iso
from .net import JSON_HEADERS, ssl_context

# Bare number words ("one", "two") are excluded deliberately: "tell me about
# the third one" contains "one" and would resolve to candidate 1 as well as 3.
ORDINALS = {
    "first": 1, "1st": 1, "top": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
}

# Maps assessment gap descriptions to the tool that can fill them.
GAP_TOOLS: dict[str, str] = {
    "no clinical interpretation": "clinvar",
    "consequence unknown": "vep",
}


# --------------------------------------------------------------------------- #
# Retrieval - deterministic
# --------------------------------------------------------------------------- #


def resolve_targets(question: str, ranked: list[Assessment]) -> list[Assessment]:
    """Work out which candidates a question is about. Never guesses silently."""
    q = question.lower()
    hits: list[Assessment] = []

    # Gene symbols are the least ambiguous handle.
    for a in ranked:
        if a.gene and re.search(rf"\b{re.escape(a.gene.lower())}\b", q):
            hits.append(a)

    # "candidate 1", "candidate 2", "#1", "rank 2"
    for m in re.finditer(r"(?:candidate|rank|#)\s*(\d+)", q):
        idx = int(m.group(1))
        if 1 <= idx <= len(ranked) and ranked[idx - 1] not in hits:
            hits.append(ranked[idx - 1])

    # "the first one", "the second"
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", q) and idx <= len(ranked):
            if ranked[idx - 1] not in hits:
                hits.append(ranked[idx - 1])

    # A comparison with nothing named means the top two - the question the
    # brief actually asks.
    if not hits and any(w in q for w in
                        ("stronger", "weaker", "better", "compare", "versus",
                         "vs", "than", "why")):
        hits = ranked[:2]

    return hits[:3]


def evidence_for(a: Assessment, rows: list[dict]) -> list[dict]:
    ids = {e for v in a.verdicts for e in v.evidence_ids}
    return [r for r in rows if r.get("evidence_id") in ids]


def build_context(targets: list[Assessment], rows: list[dict]) -> dict:
    """Everything the answer may draw on. Nothing outside this is permitted."""
    return {
        "candidates": [{
            "rank_gene": a.gene, "candidate_id": a.candidate_id,
            "strength": a.strength,
            "independent_lines": a.independent_support,
            "conflicts": a.conflicts,
            "evidence": [{
                "family": v.family, "stance": v.stance, "detail": v.detail,
                "cite": v.evidence_ids, "origin": v.derived_from,
                "weight": v.weight,
            } for v in a.verdicts],
            "circularity": a.circularity,
            "gaps": a.gaps,
        } for a in targets],
        "sources": [{
            "id": r.get("evidence_id"), "source": r.get("source"),
            "accession": r.get("record_or_accession"),
            "retrieved": r.get("retrieved_at"),
            "limitations": r.get("limitations", []),
        } for a in targets for r in evidence_for(a, rows)],
    }


# --------------------------------------------------------------------------- #
# Investigation - live tool calls when evidence is insufficient
# --------------------------------------------------------------------------- #


def parse_adhoc_variant(question: str) -> Optional[dict[str, Any]]:
    """Extract variant coordinates from a free-text question.

    Recognises two formats:
        chrom-pos-ref-alt          (candidate_id style: 10-123256215-T-G)
        chr10:123256215 T>G        (genomic notation, also T/G)
    Returns a dict with chrom/pos/ref/alt, or None.
    """
    # Format: chrom-pos-ref-alt
    m = re.search(
        r'\b(\d{1,2}|[XYM])-(\d+)-([ACGTacgt]+)-([ACGTacgt]+)\b', question)
    if m:
        return {"chrom": m.group(1), "pos": int(m.group(2)),
                "ref": m.group(3).upper(), "alt": m.group(4).upper()}
    # Format: chr10:123256215 T>G  or  10:123256215 T/G
    m = re.search(
        r'(?:chr)?(\d{1,2}|[XYM]):(\d+)\s+([ACGTacgt]+)\s*[>/]\s*([ACGTacgt]+)',
        question, re.I)
    if m:
        return {"chrom": m.group(1), "pos": int(m.group(2)),
                "ref": m.group(3).upper(), "alt": m.group(4).upper()}
    return None


def _assign_ids(rows: list[dict], prefix: str = "Q") -> None:
    """Give temporary evidence IDs to rows that have none, so citations work."""
    n = 0
    for r in rows:
        if not r.get("evidence_id"):
            n += 1
            r["evidence_id"] = f"{prefix}{n:03d}"


def investigate_adhoc(
    variant: dict[str, Any],
    tools: dict[str, Any],
    existing_rows: list[dict],
) -> tuple[list[Assessment], list[dict]]:
    """Construct a candidate from coordinates and investigate from scratch.

    This is how DNA Detective goes beyond Exomiser: a variant that Exomiser
    filtered out can be assessed on demand.
    """
    cid = (f"{variant['chrom']}-{variant['pos']}-"
           f"{variant['ref']}-{variant['alt']}")
    cand: dict[str, Any] = {
        "candidate_id": cid,
        "chrom": variant["chrom"], "pos": variant["pos"],
        "ref": variant["ref"], "alt": variant["alt"],
        "gene": "",  # VEP will fill this via side-effect
    }
    new_rows: list[dict] = []
    # VEP first — it fills cand["consequence"] and cand["gene"]
    for tool_name in ("vep", "clinvar"):
        if tool_name not in tools:
            continue
        try:
            evidence = tools[tool_name](cand)
            for ev in evidence:
                d = ev.to_dict() if hasattr(ev, "to_dict") else ev
                new_rows.append(d)
            gene_label = cand.get("gene") or cid
            print(f"  Q&A investigate: {tool_name} on {gene_label} "
                  f"-> {len(evidence)} row(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Q&A investigate: {tool_name} on {cid} failed: {exc}")

    _assign_ids(new_rows, prefix="Q")
    all_rows = existing_rows + new_rows
    assessment = assess(cand, all_rows)
    return [assessment], new_rows


def fill_gaps(
    targets: list[Assessment],
    cands: dict[str, dict],
    tools: dict[str, Any],
) -> list[dict]:
    """Call tools to fill evidence gaps for target candidates before answering.

    Only calls tools that are mapped in GAP_TOOLS and available in ``tools``.
    Returns the new evidence rows (as dicts).
    """
    extra: list[dict] = []
    for a in targets:
        if a.candidate_id not in cands:
            continue
        cand = cands[a.candidate_id]
        called: set[str] = set()
        for gap in a.gaps:
            for pattern, tool_name in GAP_TOOLS.items():
                if (pattern in gap
                        and tool_name in tools
                        and tool_name not in called):
                    called.add(tool_name)
                    try:
                        evidence = tools[tool_name](cand)
                        for ev in evidence:
                            d = (ev.to_dict()
                                 if hasattr(ev, "to_dict") else ev)
                            extra.append(d)
                        print(f"  Q&A gap-fill: {tool_name} on {a.gene} "
                              f"-> {len(evidence)} row(s)")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! Q&A gap-fill: {tool_name} on {a.gene} "
                              f"failed: {exc}")
    _assign_ids(extra, prefix="G")
    return extra


# --------------------------------------------------------------------------- #
# Templated answer - no model required
# --------------------------------------------------------------------------- #


def template_answer(targets: list[Assessment]) -> str:
    if not targets:
        return ("I could not tell which candidate that question is about. Name a "
                "gene, or say 'candidate 1'.")

    if len(targets) == 1:
        a = targets[0]
        out = [f"**{a.gene}** (`{a.candidate_id}`) - evidence strength {a.strength} "
               f"from {a.independent_support} independent line(s)."]
        for v in a.verdicts:
            if v.stance in ("supports", "conflicts"):
                cite = f" [{', '.join(v.evidence_ids)}]" if v.evidence_ids else ""
                verb = "Supporting" if v.stance == "supports" else "Against"
                out.append(f"- {verb} ({v.family}): {v.detail}{cite}")
        if a.circularity:
            out.append("\nCircularity: " + " ".join(a.circularity))
        if a.gaps:
            out.append(f"\nUnresolved: {'; '.join(a.gaps)}")
        return "\n".join(out)

    a, b = targets[0], targets[1]
    out = [f"**{a.gene} ranks above {b.gene}** - evidence strength {a.strength} "
           f"vs {b.strength}, from {a.independent_support} vs "
           f"{b.independent_support} independent lines.\n"]

    by_family_b = {v.family: v for v in b.verdicts}
    for va in a.verdicts:
        vb = by_family_b.get(va.family)
        if not vb or (va.stance == vb.stance and va.weight == vb.weight):
            continue
        ca = f" [{', '.join(va.evidence_ids)}]" if va.evidence_ids else ""
        cb = f" [{', '.join(vb.evidence_ids)}]" if vb.evidence_ids else ""
        out.append(f"- **{va.family}**: {a.gene} {va.stance} - {va.detail}{ca} "
                   f"| {b.gene} {vb.stance} - {vb.detail}{cb}")

    if b.circularity:
        out.append(f"\nNote on {b.gene}: " + " ".join(b.circularity))
    if a.circularity:
        out.append(f"\nNote on {a.gene}: " + " ".join(a.circularity))
    out.append(f"\nUnresolved for {a.gene}: {'; '.join(a.gaps) or 'none'}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Model phrasing, with citation verification
# --------------------------------------------------------------------------- #


CITE_RE = re.compile(r"\b([EACGQ]\d{3})\b")


def verify_citations(answer: str, valid: set[str]) -> tuple[str, list[str]]:
    """Strip any evidence ID that is not in the log. Returns (answer, removed).

    A citation that looks real and is not is the single most damaging thing this
    system could output.
    """
    bad = sorted({c for c in CITE_RE.findall(answer) if c not in valid})
    for c in bad:
        answer = answer.replace(c, f"[REMOVED:{c}]")
    return answer, bad


def llm_answer(question: str, context: dict, backend: str, model: str) -> Optional[str]:
    cfg = BACKENDS.get(backend, BACKENDS["ollama"])
    url, mdl = cfg["url"], model or cfg["model"]
    key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
    if cfg["key_env"] and not key:
        print(f"  ! {backend} key not set - using templated answer")
        return None

    prompt = (
        "Answer the user's question about a variant-prioritisation result.\n"
        "RULES:\n"
        "- Use ONLY the evidence below. Add no facts, no outside knowledge, no "
        "clinical advice.\n"
        "- Cite evidence IDs in square brackets, e.g. [E002]. Only IDs that "
        "appear below exist; inventing one is a serious error.\n"
        "- Do NOT state any number - frequency, score, star rating, count - "
        "unless that exact number appears in the evidence below. Quote it, do "
        "not round it, do not convert units, do not estimate.\n"
        "- If the evidence does not answer the question, say so plainly.\n"
        "- Mention any circularity and any gaps. Do not present a limitation as "
        "a strength.\n"
        "- Four sentences or fewer per candidate. Plain prose.\n\n"
        f"QUESTION: {question}\n\n"
        f"EVIDENCE: {json.dumps(context)}"
    )
    headers = dict(JSON_HEADERS)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({
        "model": mdl, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 500, "reasoning_effort": "low",
    }).encode()

    try:
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp:
            out = json.loads(resp.read().decode())
        return out["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001
            detail = ""
        print(f"  ! {backend} HTTP {exc.code}: {detail}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {backend} failed ({type(exc).__name__}) - templated answer")
        return None


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


def answer(question: str, ranked: list[Assessment], rows: list[dict],
           policy: str, backend: str, model: str, *,
           cands: Optional[dict[str, dict]] = None,
           tools: Optional[dict[str, Any]] = None) -> str:
    """Answer a follow-up question.

    Without ``cands``/``tools``: pure retrieval from stored evidence (original
    behaviour, fully backwards-compatible).

    With ``cands``/``tools``: investigation mode. The system fills evidence gaps
    before answering and can assess ad-hoc variants that were never in the
    Exomiser shortlist.
    """
    targets = resolve_targets(question, ranked)

    # --- Investigation mode --------------------------------------------------
    if cands is not None and tools:
        # Ad-hoc variant: the question names a variant not in the ranked list.
        # Investigate it from scratch — this is how DNA Detective goes beyond
        # Exomiser's shortlist.
        if not targets:
            variant = parse_adhoc_variant(question)
            if variant:
                cid = (f"{variant['chrom']}-{variant['pos']}-"
                       f"{variant['ref']}-{variant['alt']}")
                # If these coordinates match a shortlisted candidate, use the
                # existing (richer) assessment instead of building a weaker one
                existing = [a for a in ranked if a.candidate_id == cid]
                if existing:
                    targets = existing
                else:
                    targets, extra = investigate_adhoc(variant, tools, rows)
                    rows = list(rows) + extra  # copy — don't mutate caller's list

        # Gap-filling: resolved targets have gaps a tool can address. Fill them
        # before answering so the response reflects the best available evidence.
        if targets:
            extra = fill_gaps(targets, cands, tools)
            if extra:
                rows = list(rows) + extra
                # Re-assess with enriched evidence
                reassessed = []
                for a in targets:
                    if a.candidate_id in cands:
                        reassessed.append(assess(cands[a.candidate_id], rows))
                    else:
                        reassessed.append(a)
                targets = sorted(reassessed, key=lambda a: a.sort_key)
    # -------------------------------------------------------------------------

    if not targets:
        return template_answer([])

    context = build_context(targets, rows)
    # Scope citations to the rows ACTUALLY RETRIEVED for these candidates. Using
    # the whole log would let a real ID belonging to a different candidate pass
    # verification while being wrong - a plausible-looking mis-citation, which is
    # the same failure Block C's allele check exists to prevent.
    valid = {c for cand in context["candidates"]
             for ev in cand["evidence"] for c in ev["cite"]}

    if policy == "llm":
        text = llm_answer(question, context, backend, model)
        if text:
            text, bad = verify_citations(text, valid)
            if bad:
                text += ("\n\n*Citation check: " + ", ".join(bad) +
                         " do not exist in the evidence log and were removed.*")
            else:
                cited = sorted(set(CITE_RE.findall(text)))
                text += ("\n\n*Citation check: "
                         + (f"{len(cited)} ID(s) exist in this candidate's "
                            "evidence. Note: this verifies the IDs, not that "
                            "each claim matches the row it cites - check any "
                            "quoted number against the log."
                            if cited else "no evidence IDs cited.") + "*")
            return text
    return template_answer(targets)


DEMO_QUESTIONS = [
    "Why is candidate 1 stronger than candidate 2?",
    "What is wrong with COL4A1?",
    "Is the evidence for ENPP1 independent?",
    "What is still missing for FGFR2?",
]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Block G - follow-up Q&A")
    ap.add_argument("--candidates", default="outputs/blockB/candidates.json")
    ap.add_argument("--evidence", nargs="*",
                    default=["outputs/blockB/evidence.jsonl",
                             "outputs/blockC/evidence.jsonl",
                             "outputs/blockE/agent_evidence.jsonl"])
    ap.add_argument("--ask", default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="answer a fixed set of questions and save a transcript")
    ap.add_argument("--outdir", default="outputs/blockG")
    ap.add_argument("--policy", choices=["template", "llm"], default="template")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="groq")
    ap.add_argument("--model", default="")
    args = ap.parse_args(argv)

    with open(args.candidates, "r", encoding="utf-8") as fh:
        candidates = json.load(fh)
    rows = load_rows(args.evidence)
    ranked = sorted([assess(c, rows) for c in candidates], key=lambda a: a.sort_key)

    print(f"{len(candidates)} candidates, {len(rows)} evidence rows, "
          f"answering with {args.policy}")
    print("ranking: " + " > ".join(f"{i}.{a.gene}" for i, a in enumerate(ranked[:4], 1)))

    def respond(q: str) -> str:
        return answer(q, ranked, rows, args.policy, args.backend, args.model)

    if args.ask:
        print("\n" + respond(args.ask))
        return 0

    if args.demo:
        os.makedirs(args.outdir, exist_ok=True)
        lines = ["# DNA Detective - follow-up Q&A", f"\nRun: {now_iso()}",
                 f"\nAnswering mode: {args.policy}", "\n---"]
        for q in DEMO_QUESTIONS:
            print(f"\n> {q}")
            a = respond(q)
            print(a)
            lines.append(f"\n## {q}\n\n{a}\n")
        path = os.path.join(args.outdir, "qa_transcript.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwritten to {path}")
        return 0

    if args.interactive:
        print("\nAsk a follow-up question. Blank line or 'quit' to exit.\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print("\n" + respond(q) + "\n")
        return 0

    print("\nNothing asked. Use --ask, --demo or --interactive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
