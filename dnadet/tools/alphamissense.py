"""AlphaMissense pathogenicity prediction.

AlphaMissense (Cheng et al., Science 2023) uses AlphaFold protein structure
to predict missense variant pathogenicity. It is methodologically INDEPENDENT
from sequence-conservation predictors (REVEL, MVP) — concordance between
AlphaMissense and REVEL/MVP carries more weight than concordance among
sequence-based predictors alone.

This tool checks three sources in order:
    1. The VEP cache (may include ``am_pathogenicity`` and ``am_class``)
    2. The Exomiser snapshot (``effect_scores.ALPHA_MISSENSE``)
    3. Reports the gap if neither is available

No new API call is made — AlphaMissense scores ride on the existing VEP
response or the Exomiser snapshot. For ad-hoc variants, the VEP tool must
run first to populate the cache.

Usage
-----
    from dnadet.tools.alphamissense import lookup_alphamissense
    rows = lookup_alphamissense(candidate_dict)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Classification thresholds from Cheng et al., Science 2023
THRESHOLD_PATHOGENIC = 0.564
THRESHOLD_BENIGN = 0.340


def classify(score: float) -> str:
    if score >= THRESHOLD_PATHOGENIC:
        return "likely_pathogenic"
    elif score <= THRESHOLD_BENIGN:
        return "likely_benign"
    return "ambiguous"


# --------------------------------------------------------------------------- #
# VEP cache extraction
# --------------------------------------------------------------------------- #


def _check_vep_cache(
    chrom: str, pos: int, ref: str, alt: str, cache_dir: str,
) -> tuple[Optional[float], Optional[str], str]:
    """Check if the VEP cache contains AlphaMissense annotations.

    Returns (score, classification, vep_timestamp) or (None, None, "").
    """
    vep_path = os.path.join(
        cache_dir.replace("alphamissense", "vep"),
        f"vep_{chrom}_{pos}_{ref}_{alt}.json")

    if not os.path.exists(vep_path):
        return None, None, ""

    try:
        with open(vep_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None, None, ""

    body = data.get("body")
    vep_stamp = data.get("retrieved_at", "")
    if not isinstance(body, list) or not body:
        return None, None, ""

    # Search transcript consequences for AM fields
    # Prefer MANE Select, then canonical, then any with AM data
    best_score: Optional[float] = None
    best_class: Optional[str] = None

    for tc in body[0].get("transcript_consequences", []):
        am_score = tc.get("am_pathogenicity")
        am_class = tc.get("am_class")
        if am_score is not None:
            score = float(am_score)
            # Prefer MANE Select or canonical transcript
            if tc.get("mane_select") or tc.get("canonical") == 1:
                return score, am_class or classify(score), vep_stamp
            if best_score is None:
                best_score = score
                best_class = am_class or classify(score)

    return best_score, best_class, vep_stamp


# --------------------------------------------------------------------------- #
# Agent tool interface
# --------------------------------------------------------------------------- #


def lookup_alphamissense(
    cand: dict[str, Any],
    cache_dir: str = "outputs/agent_cache/alphamissense",
) -> list[Evidence]:
    """Look up AlphaMissense score for one candidate.

    Checks VEP cache first (live data), then Exomiser snapshot.
    Side-effect: ensures ``cand["effect_scores"]["ALPHA_MISSENSE"]``
    is set so ``assess()`` counts it.
    """
    cid = cand["candidate_id"]
    chrom, pos = str(cand["chrom"]), int(cand["pos"])
    ref, alt = cand["ref"], cand["alt"]
    gene = cand.get("gene") or "?"
    consequence = (cand.get("consequence") or "").lower()

    # Only meaningful for missense variants
    is_missense = "missense" in consequence

    # --- source 1: VEP cache (live) ------------------------------------------
    am_score, am_class, vep_stamp = _check_vep_cache(
        chrom, pos, ref, alt, cache_dir)

    if am_score is not None:
        # Update candidate's effect_scores
        effect_scores = cand.setdefault("effect_scores", {})
        if isinstance(effect_scores, dict):
            effect_scores["ALPHA_MISSENSE"] = round(am_score, 4)

        return [Evidence(
            evidence_id="",
            candidate_id=cid,
            category="alphamissense",
            source="AlphaMissense (via Ensembl VEP, live)",
            query=f"{chrom}:{pos} {ref}>{alt}",
            assembly="GRCh37",
            raw_field="am_pathogenicity | am_class",
            raw_value=f"{am_score:.4f} | {am_class}",
            tool_or_data_version=(
                f"AlphaMissense via Ensembl VEP, retrieved {vep_stamp}"),
            retrieved_at=vep_stamp,
            interpretation=(
                f"AlphaMissense scores {gene} {chrom}:{pos} {ref}>{alt} as "
                f"**{am_class}** (score: {am_score:.4f}). "
                f"{'Above' if am_score >= THRESHOLD_PATHOGENIC else 'Below'} "
                f"the pathogenic threshold ({THRESHOLD_PATHOGENIC}). "
                "AlphaMissense uses AlphaFold protein structure — a different "
                "methodology from REVEL/MVP (sequence conservation). "
                "Concordance between structure-based and sequence-based "
                "predictors is more meaningful than among sequence-based "
                "predictors alone."),
            limitations=[
                "AlphaMissense predicts missense pathogenicity only — it "
                "does not assess splice, frameshift, or regulatory effects.",
                "Classification thresholds are from Cheng et al., Science "
                "2023 — not calibrated for clinical use.",
            ],
        )]

    # --- source 2: Exomiser snapshot -----------------------------------------
    snapshot_score = (cand.get("effect_scores") or {}).get("ALPHA_MISSENSE")

    if snapshot_score is not None:
        am_class = classify(float(snapshot_score))
        return [Evidence(
            evidence_id="",
            candidate_id=cid,
            category="alphamissense",
            source="AlphaMissense (Exomiser 2406 snapshot)",
            query=f"{chrom}:{pos} {ref}>{alt}",
            assembly="GRCh37",
            raw_field="am_pathogenicity | am_class | source",
            raw_value=f"{snapshot_score} | {am_class} | snapshot",
            tool_or_data_version="Exomiser 2406 data bundle",
            retrieved_at="snapshot",
            interpretation=(
                f"AlphaMissense from Exomiser snapshot: {gene} scores "
                f"**{am_class}** ({snapshot_score}). "
                "This score is from the Exomiser 2406 data bundle, not a "
                "live lookup. AlphaMissense uses AlphaFold protein structure "
                "— methodologically independent from REVEL/MVP."),
            limitations=[
                "Score is from the Exomiser 2406 snapshot. A live lookup "
                "was not performed.",
                "AlphaMissense predicts missense pathogenicity only.",
            ],
        )]

    # --- no score available --------------------------------------------------
    reason = (
        f"No AlphaMissense score available for {gene} at {chrom}:{pos}."
        if is_missense else
        f"AlphaMissense is designed for missense variants; this variant's "
        f"consequence is {consequence or 'unknown'}."
    )

    return [Evidence(
        evidence_id="",
        candidate_id=cid,
        category="alphamissense",
        source="AlphaMissense — unavailable",
        query=f"{chrom}:{pos} {ref}>{alt}",
        raw_field="am_pathogenicity",
        raw_value="unavailable",
        interpretation=reason,
        limitations=[
            "AlphaMissense score not found in VEP cache or Exomiser "
            "snapshot. May not be available for this variant.",
        ],
    )]


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AlphaMissense lookup")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--cache-dir", default="outputs/agent_cache/alphamissense")
    args = ap.parse_args(argv)

    cand: dict[str, Any] = {
        "candidate_id": "cli_test",
        "chrom": args.chrom, "pos": args.pos,
        "ref": args.ref, "alt": args.alt,
        "gene": "?", "consequence": "",
        "effect_scores": {},
    }
    rows = lookup_alphamissense(cand, args.cache_dir)
    for row in rows:
        d = row.to_dict()
        print(f"\n[{d['category']}] {d['source']}")
        print(f"  raw: {d['raw_value']}")
        print(f"  interpretation: {d['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
