"""Block D (partial) - Ensembl VEP consequence annotation.

Seven of eleven candidates carry no consequence annotation in the Exomiser
output. Without knowing whether a variant is coding, splice-region, or
intergenic, it cannot be meaningfully ranked against one that IS annotated.
This tool fills that gap via the Ensembl VEP REST API.

KEY TRAP: The patient VCF is GRCh37. The default Ensembl REST host
(rest.ensembl.org) serves GRCh38 coordinates. This module uses
`grch37.rest.ensembl.org` exclusively. Querying the wrong assembly
returns a valid-looking annotation for the WRONG genomic position.

Design: same contract as clinical.py — HTTP with caching, Evidence objects,
never raises on failure, gaps are reported rather than hidden.

Usage
-----
    # As an agent tool (registered in main.py, called by the agent loop):
    from dnadet.tools.vep import annotate_candidate
    rows = annotate_candidate(candidate_dict, cache_dir="outputs/agent_cache/vep")

    # Standalone test on one variant:
    python3 -m dnadet.tools.vep --chrom 10 --pos 123256215 --ref T --alt G
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence
from ..net import ssl_context

# GRCh37 — NOT the default host.
VEP_HOST = "https://grch37.rest.ensembl.org"

# Ensembl REST allows 15 req/s unauthenticated. Be conservative.
MIN_INTERVAL = 0.25
_last_call = [0.0]

TOOL_NAME = "DNA-Detective-SummerCamp2026"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _throttle() -> None:
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


# --------------------------------------------------------------------------- #
# VCF → VEP region notation
# --------------------------------------------------------------------------- #


def vep_region_notation(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Convert VCF coordinates to Ensembl VEP region notation.

    SNV  (T>G at 100):          chrom:100-100:1/G
    Ins  (C>CGAT at 100):       chrom:101-100:1/GAT     (start > end = insertion)
    Del  (ACGT>A at 100):       chrom:101-103:1/-
    MNV  (AT>GC at 100):        chrom:100-101:1/GC
    """
    r, a = ref.upper(), alt.upper()

    # Strip shared prefix (VCF padding base)
    while r and a and r[0] == a[0]:
        r, a = r[1:], a[1:]
        pos += 1

    # Strip shared suffix
    while r and a and r[-1] == a[-1]:
        r, a = r[:-1], a[:-1]

    if not r and not a:
        # Identical after trimming — shouldn't happen in real data
        return f"{chrom}:{pos}-{pos}:1/{ref.upper()}"

    if not r:
        # Pure insertion: bases go between (pos-1) and pos
        return f"{chrom}:{pos}-{pos - 1}:1/{a}"

    if not a:
        # Pure deletion
        end = pos + len(r) - 1
        return f"{chrom}:{pos}-{end}:1/-"

    # SNV or MNV
    end = pos + len(r) - 1
    return f"{chrom}:{pos}-{end}:1/{a}"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def fetch_vep(
    chrom: str, pos: int, ref: str, alt: str,
    cache_dir: str, replay: bool = False, retries: int = 3,
) -> tuple[Optional[Any], str, str]:
    """Query Ensembl VEP REST and return (body, url, timestamp)."""
    notation = vep_region_notation(chrom, pos, ref, alt)
    url = f"{VEP_HOST}/vep/homo_sapiens/region/{notation}"
    full_url = f"{url}?content-type=application/json"

    cache_key = f"{chrom}_{pos}_{ref}_{alt}".replace("/", "_")
    cache_path = os.path.join(cache_dir, f"vep_{cache_key}.json")

    if replay or os.path.exists(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return blob.get("body"), blob.get("url", full_url), blob.get("retrieved_at", "")
        if replay:
            return None, full_url, ""

    last_err = ""
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(full_url, headers={
                "User-Agent": TOOL_NAME,
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            stamp = now_iso()
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"url": full_url, "retrieved_at": stamp, "body": body},
                          fh, indent=1)
            return body, full_url, stamp

        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After", "2")) + 0.5
                print(f"  ! VEP rate limited — waiting {wait:.0f}s")
                time.sleep(wait)
            elif exc.code == 400:
                # Bad request — the notation is wrong or the variant is
                # unrecognisable. Do not retry.
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    detail = ""
                last_err = f"HTTP 400: {detail}"
                break
            else:
                time.sleep(1.5 * (attempt + 1))

        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError, OSError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))

    print(f"  ! VEP fetch failed after {retries} tries: {last_err}")
    return None, full_url, ""


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def parse_vep(body: Any) -> dict[str, Any]:
    """Extract the most useful annotation from a VEP JSON response.

    Picks one transcript consequence in priority order:
        1. MANE Select
        2. Ensembl canonical
        3. Highest-impact transcript
    """
    if not isinstance(body, list) or not body:
        return {}

    result = body[0]
    most_severe = result.get("most_severe_consequence", "")
    tc = result.get("transcript_consequences") or []

    best: Optional[dict] = None
    for t in tc:
        if t.get("mane_select"):
            best = t
            break
    if not best:
        for t in tc:
            if t.get("canonical") == 1:
                best = t
                break
    if not best and tc:
        impact_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}
        best = min(tc, key=lambda t: impact_order.get(t.get("impact", "MODIFIER"), 3))

    if not best:
        return {"most_severe_consequence": most_severe}

    return {
        "most_severe_consequence": most_severe,
        "consequence_terms": best.get("consequence_terms", []),
        "gene_symbol": best.get("gene_symbol", ""),
        "transcript_id": best.get("transcript_id", ""),
        "biotype": best.get("biotype", ""),
        "impact": best.get("impact", ""),
        "sift_prediction": best.get("sift_prediction"),
        "sift_score": best.get("sift_score"),
        "polyphen_prediction": best.get("polyphen_prediction"),
        "polyphen_score": best.get("polyphen_score"),
        "amino_acids": best.get("amino_acids"),
        "codons": best.get("codons"),
        "mane_select": best.get("mane_select"),
        "canonical": best.get("canonical"),
    }


# --------------------------------------------------------------------------- #
# Agent tool interface: one candidate in, Evidence rows out
# --------------------------------------------------------------------------- #


def annotate_candidate(
    cand: dict,
    cache_dir: str = "outputs/agent_cache/vep",
    replay: bool = False,
) -> list[Evidence]:
    """Run VEP on one candidate and return Evidence rows.

    Side-effect: sets ``cand["consequence"]`` when the candidate has none,
    so ``assess()`` can read it on the next OBSERVE step.
    """
    cid = cand["candidate_id"]
    chrom, pos = str(cand["chrom"]), int(cand["pos"])
    ref, alt = cand["ref"], cand["alt"]
    gene = cand.get("gene") or "?"

    body, url, stamp = fetch_vep(chrom, pos, ref, alt, cache_dir, replay)

    # --- fetch failed ---------------------------------------------------------
    if body is None:
        return [Evidence(
            evidence_id="",
            candidate_id=cid,
            category="vep",
            source="Ensembl VEP (GRCh37) — FAILED",
            query=f"{chrom}:{pos} {ref}>{alt}",
            assembly="GRCh37",
            tool_or_data_version="Ensembl VEP REST (grch37.rest.ensembl.org)",
            retrieved_at=stamp or now_iso(),
            url=url,
            interpretation=(
                f"VEP query failed for {gene}. Consequence remains unknown. "
                "This gap is reported, not hidden."
            ),
            limitations=["VEP lookup failed — consequence annotation unavailable."],
        )]

    parsed = parse_vep(body)

    # --- empty annotation -----------------------------------------------------
    if not parsed.get("most_severe_consequence"):
        return [Evidence(
            evidence_id="",
            candidate_id=cid,
            category="vep",
            source="Ensembl VEP (GRCh37)",
            query=f"{chrom}:{pos} {ref}>{alt}",
            assembly="GRCh37",
            tool_or_data_version=f"Ensembl VEP REST, retrieved {stamp}",
            retrieved_at=stamp, url=url,
            interpretation=(
                f"VEP returned no consequence for {gene} at {chrom}:{pos}. "
                "The variant may be intergenic or in an unrecognised region."
            ),
            limitations=["No transcript consequence returned by VEP."],
        )]

    # --- successful annotation ------------------------------------------------
    consequence = parsed["most_severe_consequence"]
    impact = parsed.get("impact", "unknown")
    transcript = parsed.get("transcript_id", "")
    gene_vep = parsed.get("gene_symbol", "")

    # Side-effect: fill the gap in the candidate dict
    if not cand.get("consequence"):
        cand["consequence"] = consequence

    parts = [
        f"VEP annotates {gene} {chrom}:{pos} {ref}>{alt} as "
        f"**{consequence}** (impact: {impact})."
    ]
    if transcript:
        label = ("MANE Select" if parsed.get("mane_select")
                 else "canonical" if parsed.get("canonical") else "selected")
        parts.append(f"Transcript: {transcript} ({label}).")
    if parsed.get("amino_acids"):
        parts.append(f"Amino acid change: {parsed['amino_acids']}.")

    predictions = []
    if parsed.get("sift_prediction") is not None:
        predictions.append(
            f"SIFT {parsed['sift_prediction']} ({parsed.get('sift_score', '?')})")
    if parsed.get("polyphen_prediction") is not None:
        predictions.append(
            f"PolyPhen {parsed['polyphen_prediction']} "
            f"({parsed.get('polyphen_score', '?')})")
    if predictions:
        parts.append(
            f"In-silico: {'; '.join(predictions)}. Predictors share training "
            "data — agreement is not independent replication.")

    rows: list[Evidence] = [Evidence(
        evidence_id="",
        candidate_id=cid,
        category="vep",
        source="Ensembl VEP (GRCh37, live)",
        query=f"{chrom}:{pos} {ref}>{alt}",
        assembly="GRCh37",
        transcript=transcript,
        raw_field="most_severe_consequence | impact | transcript",
        raw_value=f"{consequence} | {impact} | {transcript}",
        tool_or_data_version=(
            f"Ensembl VEP REST (grch37.rest.ensembl.org), retrieved {stamp}"),
        url=url,
        retrieved_at=stamp,
        interpretation=" ".join(parts),
        limitations=[
            "Consequence enables interpretation but does not support causation — "
            "a missense change is not evidence of pathogenicity.",
            "Annotations depend on the Ensembl transcript set at query time.",
        ],
    )]

    # --- gene symbol mismatch ------------------------------------------------
    if gene_vep and gene != "?" and gene_vep != gene:
        rows.append(Evidence(
            evidence_id="",
            candidate_id=cid,
            category="vep",
            source="Ensembl VEP (GRCh37) — gene check",
            query=f"{chrom}:{pos} gene symbol",
            assembly="GRCh37",
            raw_field="gene_symbol",
            raw_value=f"Exomiser: {gene}, VEP: {gene_vep}",
            tool_or_data_version=f"Ensembl VEP REST, retrieved {stamp}",
            url=url, retrieved_at=stamp,
            interpretation=(
                f"Gene symbol mismatch: Exomiser says {gene}, VEP says "
                f"{gene_vep}. May reflect different transcript databases or "
                "an overlapping gene. Verify which annotation is current."
            ),
            limitations=[
                "Gene symbol discrepancies need manual verification against "
                "the current NCBI Gene record.",
            ],
        ))

    return rows


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="VEP consequence annotation")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--cache-dir", default="outputs/vep_cache")
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args(argv)

    notation = vep_region_notation(args.chrom, args.pos, args.ref, args.alt)
    print(f"VEP notation: {notation}")

    cand = {
        "candidate_id": "cli_test",
        "chrom": args.chrom, "pos": args.pos,
        "ref": args.ref, "alt": args.alt,
        "gene": "?",
    }
    rows = annotate_candidate(cand, args.cache_dir, args.replay)
    for row in rows:
        d = row.to_dict()
        print(f"\n[{d['category']}] {d['source']}")
        print(f"  raw: {d['raw_value']}")
        print(f"  interpretation: {d['interpretation']}")
    if cand.get("consequence"):
        print(f"\ncandidate consequence updated: {cand['consequence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
