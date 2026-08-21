"""Block E - the agent loop.

Exomiser already produced the correct ranking, so the ranking is not the
deliverable. This module produces a ranking THIS SYSTEM can defend, from cited
evidence, and says out loud where its sources disagree, where data is missing,
and where several pieces of apparent corroboration all trace back to one origin.

Three commitments shape the code
--------------------------------
1. NO FACT ENTERS EXCEPT THROUGH A TOOL. The loop never reads a database. Every
   fact arrives as an Evidence row with a source, an accession and a timestamp,
   so any claim in the final answer can be traced to a row ID.

2. INDEPENDENT SOURCES, NOT EVIDENCE ROWS. ENPP1 carries a ClinVar
   classification, PVS1+PS1 ACMG criteria, and an Exomiser filter whitelist.
   Three rows - all derived from ONE 1-star ClinVar submission. Counting three
   would be self-deception, so same-origin evidence collapses into one family
   and the collapse is reported. This is the single most important thing here.

3. `combinedScore` IS NEVER AN ANSWER. It is reproducible only with the ACMG
   posterior as a third input, so quoting it would cite a number whose
   derivation the system cannot show. Component evidence is cited instead.

The loop
--------
    OBSERVE     summarise candidates and the evidence held so far
    PRIORITIZE  pick the candidate whose uncertainty most affects the RANKING
    INVESTIGATE call exactly one tool, append its Evidence rows
    COMPARE     re-rank; note support, conflict and gaps
    REPLAN/STOP stop when the top two are separated by independent evidence and
                no remaining tool would plausibly reorder them

Policies are pluggable. The default is deterministic so the demo runs offline
with no API key; `--policy llm` hands the same decision to Kimi.

Usage
-----
    python3 -m dnadet.agent
    python3 -m dnadet.agent --policy llm --max-steps 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .contract import HPO_TERMS, Evidence
from .net import JSON_HEADERS, ssl_context

# --------------------------------------------------------------------------- #
# Evidence families
# --------------------------------------------------------------------------- #

# A "family" is an INDEPENDENT line of argument. Two rows in the same family are
# one argument stated twice, not two arguments.
FAMILIES = ("clinical", "phenotype", "population", "computational", "consequence", "literature")

CATEGORY_TO_FAMILY = {
    "clinvar": "clinical",
    "clingen": "clinical",
    "phenotype": "phenotype",
    "gnomad": "population",
    "effect": "computational",
    "vep": "consequence",
    "pubmed": "literature",
    "spliceai": "computational",
    "alphamissense": "computational",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Verdict:
    """One family's reading of one candidate."""

    family: str
    stance: str          # supports | conflicts | absent | unusable
    detail: str
    evidence_ids: list[str] = field(default_factory=list)
    derived_from: Optional[str] = None   # origin, for independence collapsing
    weight: int = 0      # evidential strength, NOT a probability


# Weights are ordinal, not calibrated. They encode one claim: expert-reviewed
# clinical evidence and a real phenotype match are worth more than a prediction.
# Counting families equally let a VUS with one AlphaMissense score outrank a
# ClinVar-pathogenic variant, which is indefensible.
W_CLINICAL_MULTI = 3     # 2-star+, multiple submitters, no conflicts
W_CLINICAL_SINGLE = 1    # 1-star, one submitter - an opinion, not a consensus
W_PHENOTYPE_MATCH = 2    # patient's own terms in the gene's disease annotation
W_POPULATION_ABSENT = 1  # PM2 support
W_COMPUTATIONAL = 1      # in silico only, shared training data
W_LITERATURE_VARIANT = 2 # published evidence about THIS specific variant
W_LITERATURE_GENE = 1    # gene-disease link in literature (not variant-specific)


@dataclass
class Assessment:
    candidate_id: str
    gene: str
    verdicts: list[Verdict]
    independent_support: int
    strength: int
    conflicts: int
    gaps: list[str]
    circularity: list[str]
    reason: str

    @property
    def sort_key(self) -> tuple:
        # Weighted evidential strength first; then how many INDEPENDENT lines
        # produced it; then fewer conflicts. Gap count is deliberately NOT a
        # tiebreak - a candidate nobody has investigated has few recorded gaps,
        # and rewarding that would rank ignorance above scrutiny.
        return (-self.strength, -self.independent_support, self.conflicts)


# --------------------------------------------------------------------------- #
# Reading the evidence
# --------------------------------------------------------------------------- #


def assess(cand: dict, rows: list[dict]) -> Assessment:
    """Turn a candidate plus its evidence rows into a defensible position."""
    cid = cand["candidate_id"]
    mine = [r for r in rows if r.get("candidate_id") == cid]
    by_family: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    for r in mine:
        fam = CATEGORY_TO_FAMILY.get(r.get("category", ""), None)
        if fam:
            by_family[fam].append(r)

    verdicts: list[Verdict] = []
    circularity: list[str] = []
    gaps: list[str] = list(cand.get("missing_evidence") or [])

    # --- clinical -----------------------------------------------------------
    label = (cand.get("clinvar_classification") or "").upper()
    stars = cand.get("clinvar_review_stars")
    clin_ids = [r["evidence_id"] for r in by_family["clinical"]]
    live_checked = any("live" in (r.get("source") or "").lower()
                       for r in by_family["clinical"])
    if not label:
        verdicts.append(Verdict("clinical", "absent",
                                "No ClinVar interpretation applies to this variant.",
                                clin_ids))
        gaps.append("no clinical interpretation")
    elif "CONFLICT" in label:
        verdicts.append(Verdict("clinical", "conflicts",
                                f"ClinVar submissions disagree ({label}).", clin_ids,
                                derived_from="clinvar"))
    elif label.startswith("PATHOGENIC") or label.startswith("LIKELY_PATHOGENIC"):
        strength = "multi-submitter" if (stars or 0) >= 2 else "single submitter"
        verdicts.append(Verdict(
            "clinical", "supports",
            f"ClinVar {label} ({stars if stars is not None else '?'} star, {strength})"
            + (", confirmed live" if live_checked else ", snapshot only"),
            clin_ids, derived_from="clinvar",
            weight=W_CLINICAL_MULTI if (stars or 0) >= 2 else W_CLINICAL_SINGLE))
        if (stars or 0) < 2:
            gaps.append("clinical support rests on a single submitter")
    elif "BENIGN" in label:
        verdicts.append(Verdict("clinical", "conflicts",
                                f"ClinVar reports {label}.", clin_ids,
                                derived_from="clinvar"))
    else:
        verdicts.append(Verdict("clinical", "unusable",
                                f"ClinVar reports {label} - carries no directional "
                                "weight.", clin_ids, derived_from="clinvar"))

    # --- phenotype ----------------------------------------------------------
    pm = cand.get("phenotype_match") or {}
    ph_ids = [r["evidence_id"] for r in by_family["phenotype"]]
    overlap_rows = [r for r in by_family["phenotype"]
                    if "patient HPO terms appear" in (r.get("interpretation") or "")]
    matched = 0
    for r in overlap_rows:
        head = (r.get("raw_value") or "")
        if "matched" in head:
            try:
                matched = max(matched, int(head.split("matched")[1].split("/")[0]))
            except (ValueError, IndexError):
                pass
    if pm.get("is_exomiser_default"):
        verdicts.append(Verdict("phenotype", "absent",
                                "Phenotype score is Exomiser's placeholder for 'no "
                                "known gene-disease relationship' - not a match.",
                                ph_ids))
        gaps.append("no phenotype evidence")
    elif matched > 0:
        verdicts.append(Verdict("phenotype", "supports",
                                f"{matched}/6 patient HPO terms appear in the gene's "
                                "disease annotation.", ph_ids, derived_from="hpo",
                                weight=W_PHENOTYPE_MATCH))
    elif ph_ids:
        verdicts.append(Verdict("phenotype", "conflicts",
                                "Gene has a disease annotation, but ZERO of the six "
                                "patient terms appear in it. Score comes from "
                                "protein-interaction channels, not phenotype match.",
                                ph_ids, derived_from="hpo"))
    else:
        verdicts.append(Verdict("phenotype", "absent", "No phenotype evidence.", []))
        gaps.append("no phenotype evidence")

    # --- population ---------------------------------------------------------
    pop_ids = [r["evidence_id"] for r in by_family["population"]]
    af = cand.get("gnomad_af")
    absent_rows = [r for r in by_family["population"]
                   if "Not observed" in (r.get("interpretation") or "")]
    if af is None and absent_rows:
        verdicts.append(Verdict("population", "supports",
                                "Absent from population databases - consistent with "
                                "PM2 for a rare dominant disorder.", pop_ids,
                                derived_from="gnomad",
                                weight=W_POPULATION_ABSENT))
        gaps.append("absence confirmed only against the 2406 snapshot")
    elif af is not None:
        verdicts.append(Verdict("population", "conflicts" if af > 2.0 else "unusable",
                                f"Observed in controls at {af}% - PM2 cannot be "
                                "claimed. Rarity alone is not evidence of "
                                "pathogenicity.", pop_ids, derived_from="gnomad"))
    else:
        verdicts.append(Verdict("population", "absent",
                                "No population data.", pop_ids))
        gaps.append("no population frequency")

    # --- computational ------------------------------------------------------
    scores = cand.get("effect_scores") or {}
    comp_ids = [r["evidence_id"] for r in by_family["computational"]]
    high = [k for k, v in scores.items() if isinstance(v, (int, float)) and v >= 0.8]
    if not scores:
        verdicts.append(Verdict("computational", "absent",
                                "No effect predictions available.", comp_ids))
        gaps.append("no effect predictors")
    elif high:
        verdicts.append(Verdict("computational", "supports",
                                f"{len(high)} predictor(s) above 0.8 ({', '.join(high)}). "
                                "Predictors share training data - agreement is not "
                                "independent replication.", comp_ids,
                                derived_from="insilico",
                                weight=W_COMPUTATIONAL))
    else:
        verdicts.append(Verdict("computational", "conflicts",
                                "Predictors do not call this damaging.", comp_ids,
                                derived_from="insilico"))

    # --- consequence --------------------------------------------------------
    cons = cand.get("consequence")
    cons_ids = [r["evidence_id"] for r in by_family["consequence"]]
    clinvar_sourced = any("ClinVar" in (r.get("source") or "")
                          for r in by_family["consequence"])
    if not cons:
        verdicts.append(Verdict("consequence", "absent",
                                "No consequence annotation - the variant's molecular "
                                "effect is unknown. It may not even be coding.",
                                cons_ids))
        gaps.append("consequence unknown (requires VEP)")
    else:
        # Consequence is a PREREQUISITE for interpretation, not evidence of
        # causation. A missense change is not a reason to believe a variant is
        # pathogenic - it is what makes the question askable at all. Recorded as
        # informational so it never inflates support.
        verdicts.append(Verdict("consequence", "unusable",
                                f"Consequence: {cons} (informational - enables "
                                "interpretation, does not support causation).",
                                cons_ids,
                                derived_from="clinvar" if clinvar_sourced else "vep",
                                weight=0))
        if clinvar_sourced:
            circularity.append(
                "Consequence is taken from the ClinVar record itself, so it is not "
                "independent of the ClinVar classification.")

    # --- literature ---------------------------------------------------------
    lit_ids = [r["evidence_id"] for r in by_family.get("literature", [])]
    lit_rows = by_family.get("literature", [])
    if lit_rows:
        raw = (lit_rows[0].get("raw_value") or "").split(" | ")
        try:
            variant_pubs = int(raw[0]) if raw else 0
        except ValueError:
            variant_pubs = 0
        try:
            gene_disease_pubs = int(raw[1]) if len(raw) > 1 else 0
        except ValueError:
            gene_disease_pubs = 0
        try:
            gene_path_pubs = int(raw[2]) if len(raw) > 2 else 0
        except ValueError:
            gene_path_pubs = 0

        if variant_pubs > 0:
            verdicts.append(Verdict("literature", "supports",
                                    f"{variant_pubs} publication(s) directly reference "
                                    "this variant — published, independent evidence.",
                                    lit_ids, derived_from="pubmed",
                                    weight=W_LITERATURE_VARIANT))
        elif gene_disease_pubs > 5:
            verdicts.append(Verdict("literature", "supports",
                                    f"Gene-disease link well-documented "
                                    f"({gene_disease_pubs} publications). Not "
                                    "variant-specific.", lit_ids,
                                    derived_from="pubmed",
                                    weight=W_LITERATURE_GENE))
        elif gene_disease_pubs > 0:
            verdicts.append(Verdict("literature", "supports",
                                    f"Gene-disease link documented "
                                    f"({gene_disease_pubs} publication(s)). "
                                    "Limited — not a widely studied association.",
                                    lit_ids, derived_from="pubmed",
                                    weight=W_LITERATURE_GENE))
        elif gene_path_pubs > 0:
            verdicts.append(Verdict("literature", "unusable",
                                    f"{gene_path_pubs} publication(s) on pathogenic "
                                    f"variants in this gene, but none link it to the "
                                    "candidate disease.", lit_ids))
        else:
            verdicts.append(Verdict("literature", "absent",
                                    "No relevant publications found.", lit_ids))
            gaps.append("no literature support for gene-disease link")
    else:
        verdicts.append(Verdict("literature", "absent",
                                "No literature search performed.", []))
        gaps.append("no literature evidence")

    # --- independence -------------------------------------------------------
    supporting = [v for v in verdicts if v.stance == "supports"]
    origins = {v.derived_from for v in supporting if v.derived_from}
    independent = len(origins)

    # Same-origin evidence contributes ONCE, at its strongest. Three rows all
    # tracing to one ClinVar submission are one argument, not three.
    strength = sum(
        max((v.weight for v in supporting if v.derived_from == origin), default=0)
        for origin in origins
    )

    clinvar_backed = [v.family for v in supporting if v.derived_from == "clinvar"]
    if len(clinvar_backed) > 1:
        circularity.append(
            f"{len(clinvar_backed)} supporting families ({', '.join(clinvar_backed)}) "
            "all derive from the same ClinVar record - counted once."
        )
    if any("whitelisted" in t for t in (cand.get("filter_trail") or [])):
        circularity.append(
            "Exomiser whitelisted this variant on the strength of its ClinVar record, "
            "so surviving the filter cascade is not evidence independent of ClinVar."
        )

    conflicts = sum(1 for v in verdicts if v.stance == "conflicts")
    reason = (
        f"evidence strength {strength} from {independent} independent "
        f"supporting line(s): "
        + "; ".join(f"{v.family} ({v.detail.split('. ')[0].rstrip('.')})"
                    for v in supporting)
        if supporting else "No supporting evidence from any independent family."
    )

    return Assessment(cid, cand.get("gene") or "?", verdicts, independent,
                      strength, conflicts, sorted(set(gaps)), circularity, reason)


# --------------------------------------------------------------------------- #
# Tools - the only way a fact enters the loop
# --------------------------------------------------------------------------- #


@dataclass
class Action:
    kind: str                     # investigate | stop
    candidate_id: str = ""
    tool: str = ""
    reason: str = ""


ToolFn = Callable[[dict], list[Evidence]]


def tool_not_implemented(name: str, what: str, blocking: str) -> ToolFn:
    """A tool that honestly reports it does not exist yet.

    The loop still calls it, still logs the gap, and the gap still appears in the
    final answer - so a missing capability is visible in the output rather than
    silently ranked around.
    """

    def _fn(cand: dict) -> list[Evidence]:
        return [Evidence(
            evidence_id="",
            candidate_id=cand["candidate_id"],
            category=name,
            source=f"{what} (NOT IMPLEMENTED)",
            query=f"{cand['chrom']}:{cand['pos']} {cand['ref']}>{cand['alt']}",
            tool_or_data_version="Block D not built",
            retrieved_at=now_iso(),
            interpretation=(
                f"The agent requested {what} for this candidate and no such tool "
                f"exists yet. {blocking}"
            ),
            limitations=[f"GAP: {what} unavailable. Conclusion is provisional."],
        )]

    return _fn


TOOLS: dict[str, ToolFn] = {
    "vep": tool_not_implemented(
        "vep", "Ensembl VEP consequence annotation",
        "Ranking of this candidate cannot be completed without knowing whether the "
        "variant is even coding."),
    "gnomad": tool_not_implemented(
        "gnomad", "live gnomAD frequency lookup",
        "PM2 currently rests on a two-year-old snapshot."),
    "splice": tool_not_implemented(
        "splice", "SpliceAI splice-effect prediction",
        "Applicable only to splice-region candidates."),
}


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


class DeterministicPolicy:
    """Rule-based. Picks the gap that most affects the ORDER, not the top score."""

    name = "deterministic"

    def propose(self, assessments: list[Assessment], cands: dict[str, dict],
                done: set[tuple[str, str]]) -> Action:
        ranked = sorted(assessments, key=lambda a: a.sort_key)

        # 0. Clinical evidence is the strongest signal (weight 3). A candidate
        #    at the ranking boundary with no clinical interpretation should be
        #    investigated first — ClinVar can move it more than any other tool.
        if "clinvar" in TOOLS:
            for a in ranked:
                if (any("no clinical interpretation" in g for g in a.gaps)
                        and (a.candidate_id, "clinvar") not in done):
                    return Action(
                        "investigate", a.candidate_id, "clinvar",
                        f"{a.gene} has no clinical interpretation — ClinVar is "
                        f"the strongest evidence family (weight {W_CLINICAL_MULTI}).")

        # 1. The decision that matters is the boundary between rank 1 and rank 2.
        #    Resolve gaps there before anything else.
        for a in ranked[:2]:
            for gap, tool in (("consequence unknown", "vep"),
                              ("absence confirmed only", "gnomad")):
                if any(gap in g for g in a.gaps) and (a.candidate_id, tool) not in done:
                    return Action("investigate", a.candidate_id, tool,
                                  f"{a.gene} sits at the rank-1/rank-2 boundary and "
                                  f"has an unresolved gap: {gap}.")

        # 2. A candidate with no consequence annotation cannot be ranked at all.
        for a in ranked:
            if any("consequence unknown" in g for g in a.gaps) and \
                    (a.candidate_id, "vep") not in done:
                return Action("investigate", a.candidate_id, "vep",
                              f"{a.gene} has no consequence annotation; it cannot be "
                              "placed relative to the others.")

        # 3. Splice models only where applicable - item 8 is a decision, not a step.
        for a in ranked:
            cons = (cands[a.candidate_id].get("consequence") or "").upper()
            if "SPLICE" in cons and (a.candidate_id, "splice") not in done:
                return Action("investigate", a.candidate_id, "splice",
                              f"{a.gene} is a splice-region variant - a splice model "
                              "is applicable here and nowhere else in the shortlist.")

        # 4. Literature - independent of ClinVar; most useful when clinical
        #    support is weak or absent for a top candidate.
        if "pubmed" in TOOLS:
            for a in ranked:
                if (any("no literature" in g for g in a.gaps)
                        and (a.candidate_id, "pubmed") not in done):
                    return Action(
                        "investigate", a.candidate_id, "pubmed",
                        f"{a.gene} has no literature evidence — PubMed can "
                        "independently confirm the gene-disease link.")

        return Action("stop", reason=stop_reason(ranked))
    


BACKENDS = {
    # Local, free, offline. Nothing leaves the machine and the demo cannot fail
    # on connectivity. Use a 7-8B model - a 0.6B will not emit reliable JSON.
    "ollama": {"url": "http://localhost:11434/v1/chat/completions",
               "model": "qwen2.5:7b", "key_env": ""},
    # Hosted open models, free tier, fast. Needs network at demo time.
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY"},
    "kimi": {"url": "https://api.moonshot.cn/v1/chat/completions",
             "model": "kimi-k2.6", "key_env": "MOONSHOT_API_KEY"},
}


class LLMPolicy:
    """Same decision as the rules, made by a model. Any OpenAI-compatible endpoint.

    Falls back to DeterministicPolicy on any failure - missing key, model down,
    unparseable reply. A demo must never fail because a model was unavailable,
    and a policy that silently guesses is worse than one that admits it.
    """

    def __init__(self, backend: str = "ollama", model: str = "",
                 base_url: str = "") -> None:
        cfg = BACKENDS.get(backend, BACKENDS["ollama"])
        self.backend = backend
        self.url = base_url or cfg["url"]
        self.model = model or cfg["model"]
        self.key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
        self.needs_key = bool(cfg["key_env"])
        self.fallback = DeterministicPolicy()

    @property
    def name(self) -> str:
        return f"llm:{self.backend}:{self.model}"

    def propose(self, assessments, cands, done) -> Action:
        if self.needs_key and not self.key:
            print(f"  ! {self.backend} key not set - using deterministic policy")
            return self.fallback.propose(assessments, cands, done)
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        # The full assessment is ~4,700 tokens per call, which exhausts a free
        # 8,000 TPM budget in two steps. The decision only needs the SHAPE of the
        # evidence, not its prose: family/stance pairs, short gap labels, and the
        # top of the ranking, which is where the ordering is actually contested.
        ranked_state = sorted(assessments, key=lambda x: x.sort_key)
        state = [{
            "id": a.candidate_id, "gene": a.gene,
            "strength": a.strength, "independent": a.independent_support,
            "conflicts": a.conflicts,
            "evidence": {v.family: v.stance for v in a.verdicts},
            "gaps": [g[:55] for g in a.gaps[:3]],
        } for a in ranked_state[:6]]

        # Listing what was ALREADY DONE invites the model to re-propose it - it
        # did, four times. Listing what remains AVAILABLE makes an invalid
        # action impossible to express.
        available = [{"candidate_id": a.candidate_id, "tool": t}
                     for a in ranked_state[:6] for t in TOOLS
                     if (a.candidate_id, t) not in done]

        prompt = (
            "You are prioritising a rare-disease variant investigation. Choose "
            "the SINGLE next action from `available` below, or stop.\n"
            "Pick the action whose result would most change the RANKING - "
            "usually the rank-1/rank-2 boundary, or a candidate that cannot be "
            "ranked at all. Do NOT sweep every tool over every candidate.\n"
            "Stop when the top two are separated by independent evidence and no "
            "remaining call would reorder them.\n\n"
            f"available: {json.dumps(available)}\n\n"
            f"state: {json.dumps(state)}\n\n"
            'Reply with JSON only, no prose: {"kind":"investigate"|"stop",'
            '"candidate_id":"...","tool":"...","reason":"one sentence"}'
        )
        # Cloudflare fronts several of these APIs and rejects urllib's default
        # "Python-urllib/3.12" with 403 error code 1010 (banned browser
        # signature). curl works, urllib does not, purely on this header.
        headers = dict(JSON_HEADERS)
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            # gpt-oss and other reasoning models emit HIDDEN thinking tokens -
            # ~2,200 per call here against a ~440-token prompt. That, not the
            # prompt, is what exhausts a free 8,000 TPM budget. This decision
            # does not need long deliberation.
            "max_completion_tokens": 300,
            "reasoning_effort": "low",
        }).encode()

        for attempt in range(2):
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp:
                    out = json.loads(resp.read().decode())
                text = out["choices"][0]["message"]["content"]
                text = text.replace("```json", "").replace("```", "").strip()
                # Small models wrap JSON in commentary - take the object.
                if not text.startswith("{"):
                    text = text[text.find("{"): text.rfind("}") + 1]
                d = json.loads(text)
                action = Action(d.get("kind", "stop"), d.get("candidate_id", ""),
                                d.get("tool", ""), d.get("reason", ""))
                if action.kind == "investigate" and (
                        action.tool not in TOOLS or action.candidate_id not in cands
                        or (action.candidate_id, action.tool) in done):
                    print(f"  ! model proposed an invalid or repeated action "
                          f"({action.tool} on {action.candidate_id}) - using rules")
                    return self.fallback.propose(assessments, cands, done)
                return action

            except urllib.error.HTTPError as exc:
                # These APIs return a JSON body explaining the failure - 401 bad
                # key, 404 decommissioned model, 429 rate limit. Printing only
                # the exception class throws that away.
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    detail = "(no body)"
                # A rate limit is a WAIT, not a failure - and the body states
                # exactly how long. Retrying once costs seconds; falling back
                # loses the model's decision entirely.
                if exc.code == 429 and attempt < 1:
                    wait = 15.0
                    m = re.search(r"try again in ([\d.]+)s", detail)
                    if m:
                        wait = min(float(m.group(1)) + 1.0, 60.0)
                    print(f"  ! {self.backend} rate limited - waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print(f"  ! {self.backend} HTTP {exc.code}: {detail}")
                print("    falling back to deterministic policy")
                return self.fallback.propose(assessments, cands, done)

            except Exception as exc:  # noqa: BLE001 - any failure falls back
                print(f"  ! {self.backend} policy failed "
                      f"({type(exc).__name__}: {exc}) - using rules")
                return self.fallback.propose(assessments, cands, done)

        return self.fallback.propose(assessments, cands, done)


def stop_reason(ranked: list[Assessment]) -> str:
    if len(ranked) < 2:
        return "Only one candidate under consideration."
    top, second = ranked[0], ranked[1]
    margin = top.strength - second.strength
    if margin > 0:
        return (
            f"{top.gene} leads {second.gene} by {margin} point(s) of weighted "
            f"evidence ({top.strength} vs {second.strength}, from "
            f"{top.independent_support} vs {second.independent_support} independent "
            "lines). Every remaining tool call would address a gap that does not "
            "affect this ordering."
        )
    return (
        f"{top.gene} and {second.gene} are NOT separated ({top.strength} points "
        f"each). Stopping because no available "
        "tool would break the tie - this is reported as an unresolved ordering, "
        "not concealed."
    )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def run(candidates: list[dict], rows: list[dict], policy, max_steps: int,
        transcript: list[str], pace: float = 0.0) -> tuple[list[Assessment], list[dict]]:
    cands = {c["candidate_id"]: c for c in candidates}
    done: set[tuple[str, str]] = set()
    new_rows: list[dict] = []
    seq = len(rows)

    for step in range(1, max_steps + 1):
        # OBSERVE
        assessments = [assess(c, rows + new_rows) for c in candidates]
        ranked = sorted(assessments, key=lambda a: a.sort_key)
        transcript.append(f"\n### Step {step} - OBSERVE\n")
        for i, a in enumerate(ranked[:5], 1):
            transcript.append(
                f"{i}. **{a.gene}** `{a.candidate_id}` - "
                f"strength {a.strength} from {a.independent_support} independent "
                f"line(s), {a.conflicts} conflict(s), "
                f"{len(a.gaps)} gap(s)")

        # PRIORITIZE + INVESTIGATE
        if pace and step > 1:
            time.sleep(pace)   # free-tier budgets are per MINUTE, not per call
        action = policy.propose(ranked, cands, done)
        transcript.append(f"\n**PRIORITIZE** -> {action.kind} "
                          f"{action.tool} {action.candidate_id}")
        transcript.append(f"> {action.reason}")

        if action.kind == "stop":
            transcript.append("\n**STOP.**")
            print(f"  step {step}: STOP - {action.reason[:90]}")
            break

        fn = TOOLS.get(action.tool)
        if not fn or action.candidate_id not in cands:
            done.add((action.candidate_id, action.tool))
            transcript.append("> Unknown tool or candidate - action discarded.")
            continue

        produced = fn(cands[action.candidate_id])
        for ev in produced:
            seq += 1
            ev.evidence_id = f"A{seq:03d}"
            new_rows.append(ev.to_dict())
        done.add((action.candidate_id, action.tool))
        print(f"  step {step}: {action.tool} on "
              f"{cands[action.candidate_id].get('gene')} -> {len(produced)} row(s)")

        # COMPARE
        after = assess(cands[action.candidate_id], rows + new_rows)
        transcript.append(f"\n**COMPARE** - {after.gene}: {after.reason}")
        if after.gaps:
            transcript.append(f"> remaining gaps: {', '.join(after.gaps)}")

    final = sorted([assess(c, rows + new_rows) for c in candidates],
                   key=lambda a: a.sort_key)
    return final, new_rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_transcript(path: str, lines: list[str], final: list[Assessment]) -> None:
    out = ["# DNA Detective - agent transcript",
           f"\nRun: {now_iso()}",
           f"\nPatient HPO: " + ", ".join(f"{h} {n}" for h, n in HPO_TERMS),
           "\n---"] + lines
    out.append("\n---\n\n## Final ranking\n")
    for i, a in enumerate(final, 1):
        out.append(f"\n### {i}. {a.gene} `{a.candidate_id}`\n")
        out.append(f"{a.reason}\n")
        for v in a.verdicts:
            mark = {"supports": "+", "conflicts": "!", "absent": "-",
                    "unusable": "o"}[v.stance]
            ids = f" [{', '.join(v.evidence_ids)}]" if v.evidence_ids else ""
            out.append(f"- `{mark}` **{v.family}**: {v.detail}{ids}")
        if a.circularity:
            out.append("\n**Circularity:**")
            out.extend(f"- {c}" for c in a.circularity)
        if a.gaps:
            out.append(f"\n**Gaps:** {'; '.join(a.gaps)}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  ! missing {p} - continuing without it")
            continue
        with open(p, "r", encoding="utf-8") as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Block E - agent loop")
    ap.add_argument("--candidates", default="outputs/blockB/candidates.json")
    ap.add_argument("--evidence", nargs="*",
                    default=["outputs/blockB/evidence.jsonl",
                             "outputs/blockC/evidence.jsonl"])
    ap.add_argument("--outdir", default="outputs/blockE")
    ap.add_argument("--policy", choices=["deterministic", "llm"],
                    default="deterministic")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ollama",
                    help="LLM backend when --policy llm (default: local Ollama)")
    ap.add_argument("--model", default="", help="override the backend's model")
    ap.add_argument("--base-url", default="", help="override the endpoint URL")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to wait between steps (free-tier TPM budgets "
                         "are per minute; try --pace 8 on Groq free)")
    args = ap.parse_args(argv)

    with open(args.candidates, "r", encoding="utf-8") as fh:
        candidates = json.load(fh)
    rows = load_rows(args.evidence)
    print(f"{len(candidates)} candidates, {len(rows)} evidence rows")

    policy = (DeterministicPolicy() if args.policy == "deterministic"
              else LLMPolicy(args.backend, args.model, args.base_url))
    print(f"policy: {policy.name}")
    transcript: list[str] = []
    final, new_rows = run(candidates, rows, policy, args.max_steps, transcript,
                          args.pace)

    os.makedirs(args.outdir, exist_ok=True)
    write_transcript(os.path.join(args.outdir, "transcript.md"), transcript, final)
    with open(os.path.join(args.outdir, "ranked.json"), "w", encoding="utf-8") as fh:
        json.dump([{
            "rank": i, "candidate_id": a.candidate_id, "gene": a.gene,
            "strength": a.strength,
            "independent_support": a.independent_support, "conflicts": a.conflicts,
            "gaps": a.gaps, "circularity": a.circularity, "reason": a.reason,
            "verdicts": [{"family": v.family, "stance": v.stance,
                          "detail": v.detail, "evidence_ids": v.evidence_ids}
                         for v in a.verdicts],
        } for i, a in enumerate(final, 1)], fh, indent=2)
    with open(os.path.join(args.outdir, "agent_evidence.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(r) + "\n")

    print("\nfinal ranking:")
    for i, a in enumerate(final[:5], 1):
        print(f"  {i}. {a.gene:<8} strength={a.strength} indep={a.independent_support} "
              f"conflicts={a.conflicts} gaps={len(a.gaps)}"
              + ("  [CIRCULARITY]" if a.circularity else ""))
    print(f"\nwritten to {args.outdir}/ (transcript.md is the demo artefact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
