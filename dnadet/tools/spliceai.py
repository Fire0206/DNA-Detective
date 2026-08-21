"""SpliceAI splice-effect prediction via the Broad Institute lookup API.

SpliceAI predicts how variants affect pre-mRNA splicing. It produces delta
scores for four splice effects:
    DS_AG (acceptor gain),  DS_AL (acceptor loss),
    DS_DG (donor gain),     DS_DL (donor loss)

Each score ranges 0-1. Thresholds (Jaganathan et al., Cell 2019):
    >= 0.2  low confidence splice-altering
    >= 0.5  moderate confidence
    >= 0.8  high confidence

SpliceAI uses a DIFFERENT training paradigm from REVEL/MVP/AlphaMissense
(pre-mRNA sequence context vs. protein features), so concordance between
SpliceAI and missense predictors carries more weight than concordance
among missense predictors alone.

Usage
-----
    # As an agent tool (registered in main.py, replaces the stub):
    from dnadet.tools.spliceai import predict_splice
    rows = predict_splice(candidate_dict)

    # Standalone test:
    python3 -m dnadet.tools.spliceai --chrom 6 --pos 132203615 --ref G --alt A
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence
from ..net import ssl_context

SPLICEAI_API = "https://spliceailookup-api.broadinstitute.org/spliceai/"

MIN_INTERVAL = 0.5
_last_call = [0.0]

TOOL_NAME = "DNA-Detective-SummerCamp2026"

# Thresholds from Jaganathan et al., Cell 2019
THRESHOLD_LOW = 0.2
THRESHOLD_MODERATE = 0.5
THRESHOLD_HIGH = 0.8


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _throttle() -> None:
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


# In-memory flag: once the API fails on a TLS/connection error in this
# process, skip all subsequent calls instantly. Resets on next run
# (new process), so when the API recovers it just works.
_api_down = [False]


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #


def fetch_spliceai(
    chrom: str, pos: int, ref: str, alt: str,
    cache_dir: str, retries: int = 3,
) -> tuple[Optional[Any], str, str]:
    """Query the Broad Institute SpliceAI lookup API."""
    variant = f"{chrom}-{pos}-{ref}-{alt}"
    params = urllib.parse.urlencode({
        "hg": "37", "distance": "50", "mask": "0", "variant": variant,
    })
    full_url = f"{SPLICEAI_API}?{params}"

    # In-memory short-circuit: API already failed this process
    if _api_down[0]:
        return None, full_url, ""

    cache_key = f"{chrom}_{pos}_{ref}_{alt}"
    cache_path = os.path.join(cache_dir, f"spliceai_{cache_key}.json")

    # Disk cache: successful lookups only
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return (blob.get("body"), blob.get("url", full_url),
                blob.get("retrieved_at", ""))

    last_err = ""
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(full_url, headers={
                "User-Agent": TOOL_NAME,
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=30,
                                        context=ssl_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            stamp = now_iso()
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"url": full_url, "retrieved_at": stamp,
                           "body": body}, fh, indent=1)
            return body, full_url, stamp

        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 429:
                time.sleep(3)
            elif exc.code == 404:
                # Variant not found — not an error, just no data
                stamp = now_iso()
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump({"url": full_url, "retrieved_at": stamp,
                               "body": {"scores": []}}, fh, indent=1)
                return {"scores": []}, full_url, stamp
            else:
                time.sleep(1.5 * (attempt + 1))

        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError, OSError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            # TLS/connection errors won't self-heal in 1.5s — stop early
            # and flag the API as down for this process
            if isinstance(exc, (urllib.error.URLError, OSError)):
                _api_down[0] = True
                break
            time.sleep(1.5 * (attempt + 1))

    print(f"  ! SpliceAI fetch failed: {last_err}")
    return None, full_url, ""


# --------------------------------------------------------------------------- #
# Fallback: VEP-based splice annotation when SpliceAI API is unreachable
# --------------------------------------------------------------------------- #


def _vep_fallback(
    cand: dict, cid: str, chrom: str, pos: int,
    ref: str, alt: str, gene: str,
    cache_dir: str, spliceai_url: str, stamp: str,
) -> list[Evidence]:
    """Use the VEP cache to report what we can about splice effects.

    The VEP tool already ran on this candidate. Its cached response tells
    us whether the variant is in a splice region (consequence label) and
    may include MaxEntScan or dbscSNV scores. SpliceAI delta scores are
    unavailable, and that gap is reported.
    """
    vep_cache = os.path.join(
        cache_dir.replace("spliceai", "vep"),
        f"vep_{chrom}_{pos}_{ref}_{alt}.json")

    consequence = ""
    vep_stamp = ""
    if os.path.exists(vep_cache):
        try:
            with open(vep_cache, "r", encoding="utf-8") as fh:
                vep_data = json.load(fh)
            vep_body = vep_data.get("body")
            vep_stamp = vep_data.get("retrieved_at", "")
            if isinstance(vep_body, list) and vep_body:
                consequence = vep_body[0].get("most_severe_consequence", "")
        except (json.JSONDecodeError, KeyError):
            pass

    parts: list[str] = [
        f"SpliceAI API (broadinstitute.org) returned no response "
        f"(connection closed during TLS handshake)."]
    if consequence and "splice" in consequence.lower():
        parts.append(
            f"VEP annotates this variant as **{consequence}**, confirming "
            "it is in a splice-relevant region.")
        parts.append(
            "However, the consequence label indicates LOCATION only — "
            "quantitative splice-effect scoring requires SpliceAI delta "
            "scores, which are unavailable.")
    elif consequence:
        parts.append(
            f"VEP consequence is {consequence} — not a splice-region "
            "variant. SpliceAI prediction is less relevant here.")
    else:
        parts.append("No VEP annotation available for fallback.")

    return [Evidence(
        evidence_id="",
        candidate_id=cid,
        category="spliceai",
        source="SpliceAI — UNREACHABLE, VEP fallback",
        query=f"{chrom}:{pos} {ref}>{alt}",
        assembly="GRCh37",
        raw_field="consequence (from VEP)",
        raw_value=consequence or "unavailable",
        tool_or_data_version=(
            f"SpliceAI API unreachable; VEP cache from {vep_stamp or '?'}"),
        url=spliceai_url or "",
        retrieved_at=stamp or now_iso(),
        interpretation=" ".join(parts),
        limitations=[
            "SpliceAI delta scores unavailable — API blocked on this "
            "network. Splice-effect QUANTIFICATION is missing; only the "
            "VEP consequence label (location) is available.",
            "This is a known infrastructure gap, not a data gap.",
        ],
    )]


# --------------------------------------------------------------------------- #
# Agent tool interface
# --------------------------------------------------------------------------- #


def predict_splice(
    cand: dict[str, Any],
    cache_dir: str = "outputs/agent_cache/spliceai",
) -> list[Evidence]:
    """Run SpliceAI on one candidate and return Evidence rows.

    Side-effect: adds ``SPLICEAI`` to ``cand["effect_scores"]`` so
    ``assess()`` counts it among computational predictors.
    """
    cid = cand["candidate_id"]
    chrom, pos = str(cand["chrom"]), int(cand["pos"])
    ref, alt = cand["ref"], cand["alt"]
    gene = cand.get("gene") or "?"

    body, url, stamp = fetch_spliceai(chrom, pos, ref, alt, cache_dir)

    # --- fetch failed: fall back to VEP cache for basic splice annotation ----
    if body is None:
        return _vep_fallback(cand, cid, chrom, pos, ref, alt, gene,
                             cache_dir, url, stamp)

    scores = body.get("scores") or []

    # --- no scores -----------------------------------------------------------
    if not scores:
        return [Evidence(
            evidence_id="",
            candidate_id=cid,
            category="spliceai",
            source="SpliceAI (Broad Institute)",
            query=f"{chrom}:{pos} {ref}>{alt}",
            assembly="GRCh37",
            tool_or_data_version=f"SpliceAI lookup, retrieved {stamp}",
            retrieved_at=stamp, url=url,
            interpretation=(
                f"SpliceAI returned no predictions for {gene} at "
                f"{chrom}:{pos}. The variant may be outside the model's "
                "coverage window."),
            limitations=["No SpliceAI scores available for this variant."],
        )]

    # --- parse scores --------------------------------------------------------
    entry = scores[0]
    ds_ag = float(entry.get("DS_AG", 0))
    ds_al = float(entry.get("DS_AL", 0))
    ds_dg = float(entry.get("DS_DG", 0))
    ds_dl = float(entry.get("DS_DL", 0))
    max_ds = max(ds_ag, ds_al, ds_dg, ds_dl)
    gene_spliceai = entry.get("SYMBOL", "")

    if max_ds >= THRESHOLD_HIGH:
        verdict = "high confidence splice-altering"
    elif max_ds >= THRESHOLD_MODERATE:
        verdict = "moderate confidence splice-altering"
    elif max_ds >= THRESHOLD_LOW:
        verdict = "low confidence splice-altering"
    else:
        verdict = "unlikely to affect splicing"

    # Side-effect: add to effect_scores for assess()
    effect_scores = cand.setdefault("effect_scores", {})
    if isinstance(effect_scores, dict):
        effect_scores["SPLICEAI"] = round(max_ds, 4)

    breakdown = (f"DS_AG={ds_ag:.4f}, DS_AL={ds_al:.4f}, "
                 f"DS_DG={ds_dg:.4f}, DS_DL={ds_dl:.4f}")

    parts: list[str] = [
        f"SpliceAI predicts {gene} {chrom}:{pos} {ref}>{alt} as "
        f"**{verdict}** (max delta score: {max_ds:.4f}).",
        f"Score breakdown: {breakdown}.",
    ]

    if max_ds >= THRESHOLD_LOW:
        effects: list[str] = []
        if ds_ag >= THRESHOLD_LOW:
            effects.append(f"acceptor gain ({ds_ag:.3f})")
        if ds_al >= THRESHOLD_LOW:
            effects.append(f"acceptor loss ({ds_al:.3f})")
        if ds_dg >= THRESHOLD_LOW:
            effects.append(f"donor gain ({ds_dg:.3f})")
        if ds_dl >= THRESHOLD_LOW:
            effects.append(f"donor loss ({ds_dl:.3f})")
        if effects:
            parts.append(f"Predicted effect(s): {', '.join(effects)}.")

    parts.append(
        "SpliceAI uses pre-mRNA sequence context — a different methodology "
        "from missense predictors (REVEL, MVP, AlphaMissense). Concordance "
        "carries more weight than among predictors sharing training data.")

    rows: list[Evidence] = [Evidence(
        evidence_id="",
        candidate_id=cid,
        category="spliceai",
        source="SpliceAI (Broad Institute, live)",
        query=f"{chrom}:{pos} {ref}>{alt}",
        assembly="GRCh37",
        raw_field="DS_AG | DS_AL | DS_DG | DS_DL | max_delta | verdict",
        raw_value=(f"{ds_ag} | {ds_al} | {ds_dg} | {ds_dl} | "
                   f"{max_ds:.4f} | {verdict}"),
        tool_or_data_version=(
            f"SpliceAI lookup API (broadinstitute.org), "
            f"retrieved {stamp}"),
        url=url,
        retrieved_at=stamp,
        interpretation=" ".join(parts),
        limitations=[
            "SpliceAI predicts splice-site effects only — it does not "
            "assess coding impact or protein function.",
            "Thresholds (0.2/0.5/0.8) are from Jaganathan et al., Cell "
            "2019 — not calibrated for clinical use.",
        ],
    )]

    # Gene symbol mismatch
    if gene_spliceai and gene != "?" and gene_spliceai != gene:
        rows.append(Evidence(
            evidence_id="",
            candidate_id=cid,
            category="spliceai",
            source="SpliceAI — gene check",
            query=f"{chrom}:{pos} gene symbol",
            raw_field="gene_symbol",
            raw_value=f"Exomiser: {gene}, SpliceAI: {gene_spliceai}",
            tool_or_data_version=f"SpliceAI lookup, retrieved {stamp}",
            url=url, retrieved_at=stamp,
            interpretation=(
                f"Gene symbol mismatch: Exomiser says {gene}, SpliceAI "
                f"says {gene_spliceai}."),
            limitations=["Gene symbol discrepancies need verification."],
        ))

    return rows


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SpliceAI prediction")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--cache-dir", default="outputs/spliceai_cache")
    args = ap.parse_args(argv)

    cand: dict[str, Any] = {
        "candidate_id": "cli_test",
        "chrom": args.chrom, "pos": args.pos,
        "ref": args.ref, "alt": args.alt,
        "gene": "?",
    }
    rows = predict_splice(cand, args.cache_dir)
    for row in rows:
        d = row.to_dict()
        print(f"\n[{d['category']}] {d['source']}")
        print(f"  raw: {d['raw_value']}")
        print(f"  interpretation: {d['interpretation']}")
    if cand.get("effect_scores", {}).get("SPLICEAI") is not None:
        print(f"\neffect_scores updated: SPLICEAI={cand['effect_scores']['SPLICEAI']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())