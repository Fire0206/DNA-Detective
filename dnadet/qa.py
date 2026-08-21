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
from typing import Any, Callable, Optional

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
    "no literature": "pubmed",
    # PM2 is assessable live now that the gnomAD lookup is implemented; without
    # these two entries the Q&A layer would leave a frequency gap it can close.
    "no population frequency": "gnomad",
    "absence confirmed only": "gnomad",
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

    # "this variant", "the variant" without naming one -> top candidate
    if not hits and re.search(
        r"\b(this|the) (variant|mutation|change)\b", q,
    ):
        hits = ranked[:1]

    # A comparison with nothing named means the top two - the question the
    # brief actually asks. But NOT if the question is about a variant being
    # absent/filtered - that should fall through to the app's handler.
    if not hits and any(w in q for w in
                        ("stronger", "weaker", "better", "compare", "versus",
                         "vs", "than", "why")):
        if not any(w in q for w in
                   ("isn't", "is not", "not in", "missing",
                    "filtered", "dropped", "removed", "excluded",
                    "why not", "where is")):
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
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[Assessment], list[dict]]:
    """Construct a candidate from coordinates and investigate from scratch.

    This is how DNA Detective goes beyond Exomiser: a variant that Exomiser
    filtered out can be assessed on demand.
    """
    _prog = progress or (lambda _: None)
    cid = (f"{variant['chrom']}-{variant['pos']}-"
           f"{variant['ref']}-{variant['alt']}")
    cand: dict[str, Any] = {
        "candidate_id": cid,
        "chrom": variant["chrom"], "pos": variant["pos"],
        "ref": variant["ref"], "alt": variant["alt"],
        "gene": "",  # VEP will fill this via side-effect
    }
    new_rows: list[dict] = []
    # VEP first — it fills cand["consequence"] and cand["gene"],
    # then ClinVar and PubMed can use that information.
    # VEP first (it supplies the gene the others search on), then the three
    # evidence families an ad-hoc variant can actually be assessed from.
    for tool_name in ("vep", "clinvar", "gnomad", "pubmed"):
        if tool_name not in tools:
            continue
        gene_label = cand.get("gene") or cid
        _prog(f"🔧 {tool_name} → {gene_label}…")
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
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Call tools to fill evidence gaps for target candidates before answering.

    Only calls tools that are mapped in GAP_TOOLS and available in ``tools``.
    Returns the new evidence rows (as dicts).
    """
    _prog = progress or (lambda _: None)
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
                    _prog(f"🔧 {tool_name} → {a.gene} "
                          f"(filling: {gap[:45]})")
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


def _explain(v: "Verdict", gene: str) -> str:
    """Translate a verdict into plain language a non-specialist can read."""
    f, s, d = v.family, v.stance, v.detail.lower()

    if f == "clinical":
        if s == "supports":
            if "multi" in d:
                return ("Multiple independent labs classify this variant "
                        "as disease-causing — the strongest clinical signal")
            return ("One lab classifies this as disease-causing, but no "
                    "independent lab has confirmed it yet")
        if s == "conflicts":
            if "benign" in d:
                return "Classified as benign or likely benign"
            return "Clinical submissions disagree on this variant"
        if s == "unusable":
            return "Clinical significance is uncertain (VUS)"
        return "No clinical classification available"

    if f == "phenotype":
        m = re.search(r"(\d+)/(\d+)", v.detail)
        if s == "supports" and m:
            return (f"Patient's symptoms match {gene}-related "
                    f"conditions ({m.group(1)} of {m.group(2)} "
                    "clinical features overlap)")
        if s == "conflicts":
            return (f"Patient's symptoms do NOT match known {gene} "
                    "conditions (0 overlap)")
        return "No phenotype data available"

    if f == "population":
        if s == "supports":
            # The cohort size belongs to whichever source answered, so take the
            # claim from the verdict rather than asserting a fixed number.
            return ("Not seen in population databases — consistent with a "
                    "rare-disease variant"
                    + (", confirmed live" if "confirmed live" in v.detail else ""))
        if s == "conflicts":
            m = re.search(r"at ([\d.]+)%", v.detail)
            return (f"Observed in healthy populations at {m.group(1)}% — "
                    "too common to cause a rare disorder" if m else
                    "Observed in healthy populations — less likely "
                    "to be disease-causing")
        if s == "unusable":
            # A frequency WAS returned; it just does not support PM2. Saying
            # "no data" here would hide a real measurement.
            m = re.search(r"at ([\d.eE+-]+)%", v.detail)
            if m:
                try:
                    pct = f"{float(m.group(1)):.4g}"
                except ValueError:
                    pct = m.group(1)
                return (f"Present in population databases at {pct}% — "
                        "rare, but PM2 requires absence")
        return "No population frequency data"

    if f == "computational":
        if s == "supports":
            m = re.search(r"(\d+) predictor", v.detail)
            n = m.group(1) if m else "Multiple"
            return (f"{n} computational tool(s) predict this variant "
                    "damages the protein")
        if s == "conflicts":
            return "Computational tools do not predict damage"
        return "No computational predictions available"

    if f == "consequence":
        m = re.search(r"Consequence: (\S+)", v.detail)
        if m:
            return (f"Variant type: {m.group(1)} (needed for "
                    "interpretation, not evidence itself)")
        return "Variant type unknown — cannot assess impact"

    if f == "literature":
        if s == "supports":
            m = re.search(r"(\d+) publication", v.detail)
            count = m.group(1) if m else "Multiple"
            if "directly reference" in d:
                return (f"{count} publication(s) directly discuss "
                        "this specific variant")
            return (f"{count} publication(s) link {gene} to "
                    "the candidate disease")
        if "not searched" in v.detail.lower():
            # Distinguish a search that returned nothing from one that never
            # ran; only the first is a finding.
            return "Literature not searched — no gene symbol available"
        if s == "unusable":
            # Publications exist for the gene but none are specific to this
            # variant or its candidate disease. Saying "none found" here would
            # understate the literature and misstate what was checked.
            m = re.search(r"(\d+) publication", v.detail)
            if m:
                return (f"{m.group(1)} publication(s) discuss {gene}, but none "
                        "are specific to this variant or its candidate disease")
        return "No relevant publications found"

    return v.detail


def _summarize(a: Assessment) -> str:
    """Plain-language clinical conclusion, not an evidence structure report."""
    gene = a.gene if a.gene and a.gene != "?" else None
    # Plain, not bolded: every sentence below already sits inside ** **,
    # and nested emphasis makes markdown close the bold at the gene name.
    gene_note = f" in {gene}" if gene else ""
    supporting = [v for v in a.verdicts if v.stance == "supports"]

    if not supporting:
        return (f"This variant{gene_note} has no supporting evidence. "
                "None of the databases or tools consulted found a reason "
                "to consider it disease-causing.")

    # Build a clinical narrative from the evidence
    has_clinical = any(v.family == "clinical" and v.stance == "supports"
                       for v in a.verdicts)
    has_multi_lab = any(v.family == "clinical" and "multi" in v.detail.lower()
                        for v in a.verdicts)
    has_phenotype = any(v.family == "phenotype" and v.stance == "supports"
                        for v in a.verdicts)
    phenotype_conflicts = any(v.family == "phenotype"
                              and v.stance == "conflicts"
                              for v in a.verdicts)
    has_population = any(v.family == "population" and v.stance == "supports"
                         for v in a.verdicts)
    has_literature = any(v.family == "literature" and v.stance == "supports"
                         for v in a.verdicts)
    lit_direct = any(v.family == "literature" and "directly" in v.detail.lower()
                     for v in a.verdicts)

    parts: list[str] = []

    # Lead with the conclusion
    if a.strength >= 7:
        parts.append(
            f"**This variant{gene_note} is very likely the cause of "
            "the patient's condition.**")
    elif a.strength >= 4:
        parts.append(
            f"**This variant{gene_note} is a plausible candidate**, "
            "but the evidence is not conclusive.")
    elif a.strength >= 2:
        parts.append(
            f"**This variant{gene_note} has weak support** and is "
            "unlikely to be the primary cause.")
    else:
        parts.append(
            f"**This variant{gene_note} has minimal evidence** "
            "linking it to the patient's condition.")

    # Explain WHY in plain language
    reasons: list[str] = []
    if has_multi_lab:
        reasons.append(
            "multiple independent genetics labs agree it is "
            "disease-causing")
    elif has_clinical:
        reasons.append(
            "one lab has classified it as disease-causing, though "
            "this has not been independently confirmed")

    if has_phenotype:
        m = None
        for v in a.verdicts:
            if v.family == "phenotype":
                m = re.search(r"(\d+)/(\d+)", v.detail)
                break
        if m:
            reasons.append(
                f"the patient's symptoms closely match known "
                f"{gene or 'associated'} conditions "
                f"({m.group(1)} of {m.group(2)} features overlap)")
    elif phenotype_conflicts:
        reasons.append(
            "however, the patient's symptoms do not match the "
            f"conditions typically associated with "
            f"{gene or 'this gene'}")

    if has_population:
        reasons.append(
            "it has not been seen in large surveys of healthy "
            "people, consistent with a rare disease variant")

    if lit_direct:
        for v in a.verdicts:
            if v.family == "literature" and "directly" in v.detail.lower():
                m = re.search(r"(\d+) publication", v.detail)
                if m:
                    reasons.append(
                        f"over {m.group(1)} published studies "
                        "directly discuss this variant")
                break
    elif has_literature:
        reasons.append(
            f"published research links {gene or 'this gene'} to "
            "the candidate disease")

    if reasons:
        parts.append(
            reasons[0][0].upper() + reasons[0][1:]
            + (", " + ", ".join(reasons[1:-1]) if len(reasons) > 2 else "")
            + (", and " + reasons[-1] if len(reasons) > 1 else "")
            + ".")

    if a.circularity:
        parts.append(
            "⚠️ Some of these evidence lines trace back to the same "
            "original source and are counted only once.")

    return " ".join(parts)


def _summarize_comparison(a: Assessment, b: Assessment) -> str:
    """Explain WHY one variant is more convincing than the other."""
    ga = a.gene if a.gene and a.gene != "?" else "the first variant"
    gb = b.gene if b.gene and b.gene != "?" else "the second variant"

    margin = a.strength - b.strength
    parts: list[str] = []

    # Lead with the conclusion
    if margin > 3:
        parts.append(
            f"**The {ga} variant is far more likely to explain this "
            f"patient's condition than {gb}.**")
    elif margin > 0:
        parts.append(
            f"**The {ga} variant has stronger support than {gb}**, "
            "though the gap is not large.")
    else:
        parts.append(
            f"**Neither variant clearly outperforms the other** — "
            "both have similar levels of evidence.")

    # Explain the key reasons
    af = {v.family: v for v in a.verdicts}
    bf = {v.family: v for v in b.verdicts}
    reasons: list[str] = []

    # Clinical
    ac, bc = af.get("clinical"), bf.get("clinical")
    if ac and bc:
        if ac.stance == "supports" and bc.stance != "supports":
            reasons.append(
                f"{ga} has clinical confirmation while {gb} does not")
        elif (ac.stance == "supports" and bc.stance == "supports"
              and ac.weight > bc.weight):
            reasons.append(
                f"{ga} has agreement from multiple independent labs "
                f"while {gb} relies on a single lab's assessment")

    # Phenotype
    ap, bp = af.get("phenotype"), bf.get("phenotype")
    if ap and bp:
        if ap.stance == "supports" and bp.stance == "conflicts":
            reasons.append(
                f"the patient's symptoms match {ga}-related conditions "
                f"but do not match {gb}")

    # Literature
    al, bl = af.get("literature"), bf.get("literature")
    if al and bl:
        if al.stance == "supports" and bl.stance != "supports":
            reasons.append(
                f"published research supports {ga} but not {gb}")

    # Semicolons, not spaces: these are full clauses and run together
    # into one unreadable sentence when joined on whitespace.
    if reasons:
        joined = "; ".join(reasons)
        parts.append(joined[0].upper() + joined[1:] + ".")

    if b.circularity and not a.circularity:
        parts.append(
            f"Additionally, {gb}'s evidence has a circularity "
            "problem — what looks like multiple independent lines "
            "actually traces back to a single source.")

    return " ".join(parts)


def _variant_label(a: Assessment) -> str:
    """'chr10:123256215 T>G (FGFR2)' - variant first, gene as context."""
    parts = a.candidate_id.split("-")
    if len(parts) >= 4:
        chrom, pos = parts[0], parts[1]
        ref, alt = parts[2], "-".join(parts[3:])
        coord = f"chr{chrom}:{pos} {ref}>{alt}"
    else:
        coord = a.candidate_id
    if a.gene and a.gene != "?":
        return f"{coord} ({a.gene})"
    return coord


def _short_label(a: Assessment) -> str:
    """Gene when known, coordinates otherwise. For table column headers."""
    return a.gene if a.gene and a.gene != "?" else a.candidate_id


def _literature_details(evidence_ids: list[str], rows: list[dict]) -> str:
    """Extract top paper titles as clickable PubMed links."""
    if not rows:
        return ""
    for eid in evidence_ids:
        row = next((r for r in rows if r.get("evidence_id") == eid), None)
        if not row or row.get("category") != "pubmed":
            continue
        interp = row.get("interpretation", "")
        # The interpretation ends with:
        # "Top results: PMID:X Author et al., Journal (Year); ..."
        m = re.search(r"Top results?: (.+?)\.?\s*$", interp)
        if not m:
            continue
        entries = m.group(1).split("; ")
        out: list[str] = []
        for entry in entries[:3]:
            pm = re.search(r"PMID:(\d+)\s*(.*)", entry)
            if pm:
                pmid, desc = pm.group(1), pm.group(2).strip()
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                out.append(f"- [{desc}]({url})")
            else:
                out.append(f"- {entry}")
        return "\n".join(out)
    return ""


def _top_publications(a: Assessment, rows: list[dict]) -> str:
    """Markdown bullet list of this candidate's top papers, or ''."""
    for v in a.verdicts:
        if v.family == "literature" and v.stance == "supports":
            return _literature_details(v.evidence_ids, rows)
    return ""


_GAP_HUMAN: dict[str, str] = {
    "absence confirmed only against the 2406 snapshot":
        "Population data is from a 2-year-old snapshot, not a live check",
    "gnomAD v4 allele frequency not yet queried live":
        "Population data is from a 2-year-old snapshot, not a live check",
    "no transcript accession in Exomiser output - MANE Select must come from VEP":
        "Transcript information came from VEP, not the original data",
    "read depth not available from Exomiser output":
        "Sequencing quality data was not available",
    "clinical support rests on a single submitter":
        "Only one lab has classified this — independent confirmation needed",
    "no clinical interpretation":
        "No genetics lab has assessed this variant yet",
    "consequence unknown (requires VEP)":
        "We don't know what this DNA change does to the protein",
    "no effect predictors":
        "No computational tools have assessed this variant",
    "no literature support for gene-disease link":
        "No published research links this gene to the disease",
    "no literature evidence":
        "No literature search was performed",
    "no phenotype evidence":
        "No symptom-matching data available",
    "no population frequency":
        "No data on how common this variant is in healthy people",
}


def _humanise_gaps(gaps: list[str]) -> str:
    """Turn technical gap labels into plain-language caveats."""
    if not gaps:
        return ""
    human: list[str] = []
    for g in gaps:
        # Try exact match first, then substring match
        h = _GAP_HUMAN.get(g)
        if not h:
            for pattern, explanation in _GAP_HUMAN.items():
                if pattern in g:
                    h = explanation
                    break
        h = h or g
        # Several technical gaps collapse to the same plain-language
        # caveat; say it once.
        if h not in human:
            human.append(h)
    return "; ".join(human)


def template_answer(targets: list[Assessment],
                    rows: Optional[list[dict]] = None) -> str:
    if not targets:
        return ("I could not tell which variant that question is about. "
                "Try a gene name (e.g. FGFR2, ENPP1) — it refers to "
                "the specific variant in that gene — or paste "
                "coordinates like `chr10:123256215 T>G`.")

    _ICON = {"supports": "✅", "conflicts": "⚠️",
             "absent": "—", "unusable": "⚪"}

    if len(targets) == 1:
        a = targets[0]
        lines = [f"### {_variant_label(a)}\n"]

        # Plain-language summary
        lines.append(_summarize(a))
        lines.append("")

        # Evidence table with explanations
        lines.append("| Family | Assessment | Evidence |")
        lines.append("|---|---|---|")
        gene_label = _short_label(a)
        for v in a.verdicts:
            icon = _ICON.get(v.stance, "?")
            cite = ", ".join(v.evidence_ids) if v.evidence_ids else "—"
            explanation = _explain(v, gene_label).replace("|", "∣")
            lines.append(
                f"| {v.family.title()} | {icon} {explanation} "
                f"| {cite} |")

        # Top publications (if literature evidence exists)
        if rows:
            papers = _top_publications(a, rows)
            if papers:
                lines.append(f"\n**Top publications:**\n{papers}")

        if a.gaps:
            lines.append(f"\n**Caveats:** {_humanise_gaps(a.gaps)}")
        return "\n".join(lines)

    # --- comparison: two candidates ------------------------------------------
    a, b = targets[0], targets[1]
    a_lbl, b_lbl = _short_label(a), _short_label(b)
    lines = [f"### {_variant_label(a)} vs {_variant_label(b)}\n"]

    # Plain-language comparison summary
    lines.append(_summarize_comparison(a, b))
    lines.append("")

    # Summary metrics
    lines.append(f"| Metric | {a_lbl} | {b_lbl} |")
    lines.append("|---|---|---|")
    lines.append(f"| Strength | **{a.strength}** | **{b.strength}** |")
    lines.append(
        f"| Independent lines | {a.independent_support} "
        f"| {b.independent_support} |")
    lines.append(f"| Conflicts | {a.conflicts} | {b.conflicts} |")

    # Per-family comparison with explanations
    by_b = {v.family: v for v in b.verdicts}
    seen: set[str] = set()

    lines.append(f"\n| Family | {a_lbl} | {b_lbl} |")
    lines.append("|---|---|---|")

    def _cell(v, gene: str) -> str:
        icon = _ICON.get(v.stance, "?")
        expl = _explain(v, gene).replace("|", "∣")
        # Truncate for table readability
        if len(expl) > 75:
            expl = expl[:72] + "…"
        cite = (f" ({', '.join(v.evidence_ids)})"
                if v.evidence_ids else "")
        return f"{icon} {expl}{cite}"

    for va in a.verdicts:
        seen.add(va.family)
        vb = by_b.get(va.family)
        cell_b = _cell(vb, b_lbl) if vb else "—"
        lines.append(
            f"| {va.family.title()} | {_cell(va, a_lbl)} "
            f"| {cell_b} |")

    for vb in b.verdicts:
        if vb.family not in seen:
            lines.append(
                f"| {vb.family.title()} | — "
                f"| {_cell(vb, b_lbl)} |")

    # Top publications for each candidate
    if rows:
        for cand, label in ((a, a_lbl), (b, b_lbl)):
            papers = _top_publications(cand, rows)
            if papers:
                lines.append(
                    f"\n**{label} — top publications:**\n{papers}")

    if a.gaps or b.gaps:
        caveats = []
        if a.gaps:
            caveats.append(f"{a_lbl}: {_humanise_gaps(a.gaps)}")
        if b.gaps:
            caveats.append(f"{b_lbl}: {_humanise_gaps(b.gaps)}")
        lines.append(f"\n**Caveats:** {' · '.join(caveats)}")

    return "\n".join(lines)


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
        "- Refer to candidates by variant coordinates with gene in "
        "parentheses, e.g. 'chr10:123256215 T>G (FGFR2)', NEVER "
        "'candidate 1'. The ranking is of VARIANTS, not genes.\n"
        "- Explain what the evidence MEANS for the patient in plain "
        "language. 'Multiple labs confirmed this variant causes disease' "
        "is better than 'ClinVar PATHOGENIC (2★, multi-submitter)'. "
        "The reader may not know what ClinVar or star ratings mean.\n"
        "- Structure your answer: use a markdown table to compare evidence "
        "families side by side when comparing candidates. Follow with brief "
        "prose for circularity and gaps.\n"
        "- After the table, write at most 3 sentences about circularity "
        "and gaps. Do NOT restate the table contents as bullet points or "
        "a 'What this means' section — the table IS the explanation.\n"
        "- Use markdown formatting only. Never use HTML tags.\n\n"
        f"QUESTION: {question}\n\n"
        f"EVIDENCE: {json.dumps(context)}"
    )
    headers = dict(JSON_HEADERS)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({
        "model": mdl, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 3000, "reasoning_effort": "low",
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


_ACRONYMS = {"MVP", "REVEL", "CADD", "SIFT", "MPC", "VEST", "GERP", "PHYLOP"}


def _pretty_source(source: str) -> str:
    """ALPHA_MISSENSE -> AlphaMissense, SPLICE_AI -> SpliceAI, REVEL -> REVEL.

    Purely structural: an ALL_CAPS underscore token is a predictor name from
    the Exomiser bundle, so it is title-cased and joined. Anything already
    written for humans is left exactly as it is.
    """
    if not (source.isupper() and source.replace("_", "").isalnum()):
        return source
    if source in _ACRONYMS:
        return source
    parts = source.split("_")
    if parts[-1] == "AI":                      # SPLICE_AI -> SpliceAI
        return "".join(w.capitalize() for w in parts[:-1]) + "AI"
    return "".join(w.capitalize() for w in parts)


def _gap_reason(source: str, searchable: bool = False) -> str:
    """Why a lookup produced nothing, read off the source label.

    Tools mark their own failures in the source string ("PubMed - skipped",
    "SpliceAI - UNREACHABLE", "... (NOT IMPLEMENTED)"), so the explanation is
    already in the data rather than in a table this file has to maintain.
    """
    low = source.lower()
    if "not implemented" in low:
        return "not queried - live lookup not implemented"
    if "skipped" in low:
        return "not searched - no gene symbol available"
    if "unreachable" in low or "failed" in low:
        return "unreachable - the service did not respond"
    if "unavailable" in low:
        return "unavailable for this variant"
    if searchable:
        # The row carries a link, so the lookup ran and the reader can open the
        # same search: it returned nothing rather than failing.
        return "no record found"
    return "not retrieved"


def _measurement(row: dict) -> str:
    """The value a computed row reports, short enough to sit inside prose.

    Scores render to 2 decimals ("0.9918" -> "0.99"); anything else that is not
    a bare JSON blob is passed through, and blobs are dropped so the reader is
    not shown raw payload.
    """
    raw = str(row.get("raw_value", "")).strip()
    if not raw or raw.startswith(("{", "[")):
        return ""
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return raw if len(raw) <= 40 else ""


def _humanise_citations(text: str, rows: list[dict]) -> str:
    """Turn raw evidence IDs into clickable links with readable labels.

    [E002] or bare E002  ->  [E002: ClinVar](url)

    A row with no url still gets the readable label, just unlinked.
    """
    if not rows:
        return text

    # Build lookup: evidence_id -> (tag, url, kind, measurement)
    lookup: dict[str, tuple] = {}
    lookup_src: dict[str, str] = {}
    for r in rows:
        eid = r.get("evidence_id", "")
        if not eid:
            continue
        source = r.get("source", "")
        url = r.get("url", "")
        cat = r.get("category") or ""
        if "ClinVar" in source:
            tag = "ClinVar"
        elif "VEP" in source:
            tag = "VEP"
        elif "PubMed" in source:
            tag = "PubMed"
        elif "SpliceAI" in source or "splice" in source.lower():
            tag = "SpliceAI"
        elif "AlphaMissense" in source:
            tag = "AlphaMissense"
        elif "phenotype" in cat:
            tag = "Phenotype"
        elif "gnomad" in cat.lower():
            tag = "gnomAD"
        elif "frequency" in cat:
            tag = "Population"
        elif "clingen" in cat.lower():
            tag = "ACMG/AMP"
        elif source:
            # Exomiser names its predictors in the data itself (ALPHA_MISSENSE,
            # REVEL, SPLICE_AI, and whatever a later bundle adds), so prettify
            # the token rather than matching against a fixed list that a new
            # dataset would silently fall through.
            tag = _pretty_source(source)
        elif cat:
            tag = cat
        else:
            tag = "evidence"
        lookup[eid] = (tag, url, r.get("record_kind", ""),
                       _measurement(r))
        lookup_src[eid] = source

    # Normalise "[E002]" to "E002" so the next step cannot nest brackets.
    text = re.sub(r"\[([EACGQ]\d{3})\]", r"\1", text)

    def _replace(m: "re.Match") -> str:
        """Render one citation as `<what it says> (<evidence id>)`.

        The id sits in the same place for every row, so `(E004)` always means
        "this claim traces to row E004 of the log". Whether the label is a link
        carries the other distinction on its own: a link means an external
        record exists to open, no link means the value was computed here and
        has no page to point at. Encoding both facts in the id's position made
        the two kinds of row look like different sorts of reference.
        """
        eid = m.group(1)
        if eid not in lookup:
            return eid
        tag, url, kind, measurement = lookup[eid]

        if kind == "gap":
            what = f"{tag} {_gap_reason(lookup_src.get(eid, ''), bool(url))}"
        elif kind == "computed":
            what = f"{tag} {measurement}" if measurement else tag
        else:
            what = tag

        if url:
            what = f"[{what}]({url})"
        return f"{what} ({eid})"

    return CITE_RE.sub(_replace, text)


def _llm_rephrase(
    summary: str, caveats: str, question: str,
    backend: str, model: str,
) -> Optional[dict[str, str]]:
    """Ask the LLM to rephrase summary + caveats in natural language.

    Returns {"summary": "...", "caveats": "..."} or None on failure.
    The LLM receives ONLY the prose to rephrase - no evidence data,
    no tables, no IDs. It cannot hallucinate facts it was never given.
    """
    cfg = BACKENDS.get(backend, BACKENDS.get("kimi", BACKENDS.get("ollama")))
    if not cfg:
        return None
    url, mdl = cfg["url"], model or cfg["model"]
    key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
    if cfg["key_env"] and not key:
        return None

    prompt = (
        "You are a geneticist explaining variant analysis results. "
        f"The user asked: \"{question}\"\n\n"
        "Rephrase the SUMMARY to directly answer their question. "
        "Emphasize the parts most relevant to what they asked - "
        "if they asked about papers, lead with literature findings; "
        "if about symptoms, lead with phenotype matching; "
        "if a general question, give the overall conclusion.\n\n"
        "RULES:\n"
        "- Do NOT change any numbers, gene names, or factual claims.\n"
        "- Do NOT add any information not in the original.\n"
        "- Do NOT use bullet points or markdown formatting.\n"
        "- Summary: 3-4 fluent sentences. Caveats: 1-2 sentences.\n"
        "- Reply ONLY in this format, nothing else:\n"
        "SUMMARY: [your rephrased summary]\n"
        "CAVEATS: [your rephrased caveats]\n\n"
        f"SUMMARY: {summary}\n\n"
        f"CAVEATS: {caveats or 'None'}"
    )

    headers = dict(JSON_HEADERS)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({
        "model": mdl, "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 500,
    }).encode()

    try:
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(
            req, timeout=30, context=ssl_context(),
        ) as resp:
            out = json.loads(resp.read().decode())
        text = out["choices"][0]["message"]["content"].strip()

        result: dict[str, str] = {}
        sm = re.search(
            r"SUMMARY:\s*(.+?)(?=\nCAVEATS:|\Z)", text, re.DOTALL)
        cm = re.search(r"CAVEATS:\s*(.+?)$", text, re.DOTALL)
        if sm:
            result["summary"] = sm.group(1).strip()
        if cm and cm.group(1).strip().lower() not in ("none", "n/a"):
            result["caveats"] = cm.group(1).strip()
        return result if result else None

    except Exception as exc:  # noqa: BLE001
        print(f"  ! LLM rephrase failed: {exc}")
        return None


def _rephrase_template(
    base: str, question: str, backend: str, model: str,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Extract prose from template answer, rephrase via LLM, reassemble.

    The table, publications, and evidence links are NEVER sent to the LLM.
    Only the summary paragraph and caveats are rephrased.
    """
    _prog = progress or (lambda _: None)

    # Find the first table row to split summary from structured content
    table_idx = base.find("\n|")
    if table_idx < 0:
        return base

    before_table = base[:table_idx]
    rest = base[table_idx:]

    # Split: header (### ...) and summary paragraph
    header_end = before_table.find("\n\n")
    if header_end < 0:
        return base
    summary = before_table[header_end:].strip()
    if not summary:
        return base

    # Extract caveats (at the end)
    caveats = ""
    caveats_match = re.search(
        r"\*\*Caveats:\*\*\s*(.+)$", rest, re.DOTALL)
    if caveats_match:
        caveats = caveats_match.group(1).strip()

    _prog("✨ Polishing language…")
    rephrased = _llm_rephrase(summary, caveats, question, backend, model)
    if not rephrased:
        return base

    # Replace summary (first occurrence only)
    if rephrased.get("summary"):
        base = base.replace(summary, rephrased["summary"], 1)

    # Replace caveats
    if rephrased.get("caveats") and caveats:
        base = base.replace(
            f"**Caveats:** {caveats}",
            f"**Caveats:** {rephrased['caveats']}",
            1)

    return base


def answer(question: str, ranked: list[Assessment], rows: list[dict],
           policy: str, backend: str, model: str, *,
           cands: Optional[dict[str, dict]] = None,
           tools: Optional[dict[str, Any]] = None,
           progress: Optional[Callable[[str], None]] = None) -> str:
    """Answer a follow-up question.

    Without ``cands``/``tools``: pure retrieval from stored evidence (original
    behaviour, fully backwards-compatible).

    With ``cands``/``tools``: investigation mode. The system fills evidence gaps
    before answering and can assess ad-hoc variants that were never in the
    Exomiser shortlist.
    """
    _prog = progress or (lambda _: None)

    _prog("🔍 Identifying candidate…")
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
                    _prog(f"🧬 New variant — investigating {cid}…")
                    targets, extra = investigate_adhoc(
                        variant, tools, rows, progress=progress)
                    rows = list(rows) + extra  # copy — don't mutate caller's list

        # Gap-filling: resolved targets have gaps a tool can address. Fill them
        # before answering so the response reflects the best available evidence.
        if targets:
            _prog(f"📋 Checking evidence for "
                  f"{', '.join(a.gene for a in targets)}…")
            extra = fill_gaps(targets, cands, tools, progress=progress)
            if extra:
                _prog(f"📊 Re-assessing with {len(extra)} new evidence…")
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
        return template_answer([], rows)

    _prog("💭 Composing answer…")
    base = template_answer(targets, rows)

    # The model no longer writes the answer - it only rephrases prose the
    # template already produced. The table, evidence links and publications
    # are never sent to it, so it cannot alter or invent a citation.
    if policy == "llm" and not base.startswith("I could not tell"):
        base = _rephrase_template(base, question, backend, model, progress)

    return _humanise_citations(base, rows)


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