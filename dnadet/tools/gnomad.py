"""Live gnomAD allele-frequency lookup via the Broad Institute GraphQL API.

gnomAD aggregates allele frequencies across large reference cohorts. For rare
disease work the question is narrow: is this variant too common in people
without the disease to be its cause? Absence supports ACMG PM2; an appreciable
frequency defeats it (BA1/BS1).

Three things this module is careful about, because each has a way of being
silently wrong:

Build.  GRCh37/hg19 coordinates belong to the `gnomad_r2_1` dataset. gnomAD v4
is GRCh38; querying it with GRCh37 coordinates returns a different locus and
therefore a frequency for a different variant. The dataset is pinned here and
never taken from the caller.

Units.  The API reports allele frequency as a FRACTION (1.19e-05). The rest of
this project stores `gnomad_af` as a PERCENT, and the assessor's common-variant
test reads `af > 2.0` as "above 2 percent". Converting at the boundary is the
whole reason this module owns the conversion rather than passing the raw value
on: a 5% variant arriving as 0.05 would be scored as vanishingly rare.

Absence vs failure.  "Variant not found" is an ANSWER - the variant is absent
from 141,456 reference genomes, which is evidence. A network error is a GAP and
means nothing at all. Reporting the second as the first would manufacture PM2
support out of a timeout, so they return different rows.

Usage
-----
    # As an agent tool (registered in main.py, replaces the stub):
    from dnadet.tools.gnomad import lookup_frequency
    rows = lookup_frequency(candidate_dict)

    # Standalone test:
    python3 -m dnadet.tools.gnomad --chrom 10 --pos 123256215 --ref T --alt G
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence
from ..net import ssl_context

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
# GRCh37 lives in r2.1.1. Pinned, never derived from the caller - see module docstring.
DATASET = "gnomad_r2_1"
GNOMAD_UI = "https://gnomad.broadinstitute.org/variant/{vid}?dataset=" + DATASET

MIN_INTERVAL = 0.5
_last_call = [0.0]

TOOL_NAME = "DNA-Detective-SummerCamp2026"

# ACMG-informed thresholds, expressed in PERCENT to match `gnomad_af`.
BA1_PERCENT = 5.0    # stand-alone benign: far too common for a rare disorder
BS1_PERCENT = 1.0    # strong benign: above any plausible rare-disease frequency

QUERY = """
query VariantFrequency($vid: String!, $ds: DatasetId!) {
  variant(variantId: $vid, dataset: $ds) {
    variant_id
    exome { ac an af }
    genome { ac an af }
  }
}
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _throttle() -> None:
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """gnomAD's variant identifier: chrom-pos-ref-alt, no 'chr' prefix."""
    return f"{str(chrom).replace('chr', '')}-{pos}-{ref}-{alt}"


def gnomad_url(vid: str) -> str:
    """Browser page for the variant, for a human following a citation."""
    return GNOMAD_UI.format(vid=vid)


def fetch_frequency(
    vid: str,
    cache_dir: str = "outputs/agent_cache/gnomad",
    retries: int = 3,
) -> tuple[Optional[dict], str, str, str]:
    """POST the GraphQL query.

    Returns (payload, url, timestamp, error). `payload` is the `variant` object,
    or None. An empty `error` with a None payload means the API answered and the
    variant is genuinely absent; a non-empty `error` means the lookup failed and
    nothing may be concluded.
    """
    body = json.dumps({
        "query": QUERY,
        "variables": {"vid": vid, "ds": DATASET},
    }).encode("utf-8")

    digest = hashlib.sha1(body).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"gnomad_{vid}_{digest}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return (blob.get("variant"), gnomad_url(vid),
                blob.get("retrieved_at", ""), blob.get("error", ""))

    last_err = ""
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(
                GNOMAD_API, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": TOOL_NAME})
            with urllib.request.urlopen(req, timeout=30,
                                        context=ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            variant = (data.get("data") or {}).get("variant")
            errors = data.get("errors") or []
            # "Variant not found" is the API's way of saying absent. Every other
            # GraphQL error is a failed lookup and must not be read as absence.
            hard_error = ""
            if variant is None and errors:
                messages = "; ".join(
                    str(e.get("message", "")) for e in errors if isinstance(e, dict))
                if "not found" not in messages.lower():
                    hard_error = messages

            stamp = now_iso()
            if not hard_error:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump({"variant": variant, "retrieved_at": stamp,
                               "error": "", "variant_id": vid}, fh, indent=1)
            return variant, gnomad_url(vid), stamp, hard_error

        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError, OSError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 * (attempt + 1))

    print(f"  ! gnomAD fetch failed after {retries} tries: {last_err}")
    return None, gnomad_url(vid), "", last_err or "lookup failed"


def combined_frequency(variant: dict) -> tuple[Optional[float], int, int, str]:
    """Pool exome and genome calls into one frequency.

    Returns (percent, allele_count, allele_number, detail). Pooling by summing
    counts rather than averaging the two frequencies is what gnomAD's own
    variant page reports, and avoids weighting a 2,000-genome callset equally
    with a 250,000-exome one.
    """
    ac = an = 0
    seen = []
    for label in ("exome", "genome"):
        block = variant.get(label)
        if isinstance(block, dict) and block.get("an"):
            ac += int(block.get("ac") or 0)
            an += int(block.get("an") or 0)
            seen.append(f"{label} {block.get('ac')}/{block.get('an')}")
    if not an:
        return None, 0, 0, ""
    # Fraction -> percent. See the module docstring: the rest of the project
    # stores and compares this value as a percentage.
    return (ac / an) * 100.0, ac, an, ", ".join(seen)


def lookup_frequency(
    cand: dict[str, Any],
    cache_dir: str = "outputs/agent_cache/gnomad",
) -> list[Evidence]:
    """Look up a candidate's allele frequency in gnomAD r2.1.1 (GRCh37).

    Side-effect: sets ``cand["gnomad_af"]`` (percent) when a frequency is found,
    which is what the assessor reads to decide PM2/BS1/BA1.
    """
    cid = cand["candidate_id"]
    chrom, pos = str(cand["chrom"]), int(cand["pos"])
    ref, alt = cand["ref"], cand["alt"]
    gene = cand.get("gene") or "?"
    vid = variant_id(chrom, pos, ref, alt)

    variant, url, stamp, error = fetch_frequency(vid, cache_dir)

    # --- lookup failed: a gap, and explicitly not an absence -----------------
    if error:
        return [Evidence(
            evidence_id="",
            record_kind="gap",
            candidate_id=cid,
            category="gnomad",
            source="gnomAD (live) - FAILED",
            query=f"variantId={vid} dataset={DATASET}",
            assembly="GRCh37",
            tool_or_data_version=f"gnomAD GraphQL API ({DATASET})",
            url=url,
            retrieved_at=stamp or now_iso(),
            interpretation=(
                f"gnomAD lookup failed for {gene} {vid} ({error}). No frequency "
                "conclusion can be drawn - this is a gap, NOT evidence that the "
                "variant is absent."
            ),
            limitations=["Live gnomAD retrieval failed; PM2 cannot be assessed "
                         "from this row."],
        )]

    # --- variant absent: an answer, and evidence for PM2 ---------------------
    if variant is None:
        return [Evidence(
            evidence_id="",
            record_kind="retrieved",
            candidate_id=cid,
            category="gnomad",
            source="gnomAD (live)",
            record_or_accession=vid,
            query=f"variantId={vid} dataset={DATASET}",
            assembly="GRCh37",
            raw_field="variant",
            raw_value="null (not present in gnomAD r2.1.1)",
            tool_or_data_version=f"gnomAD GraphQL API ({DATASET})",
            url=url,
            retrieved_at=stamp,
            # The assessor keys the PM2 verdict on this phrase.
            interpretation=(
                f"Not observed in gnomAD r2.1.1 (GRCh37), live. Consistent with "
                f"PM2 (absent from controls) for a rare dominant disorder, and "
                f"confirms the offline snapshot against current data."
            ),
            limitations=[
                "Absence from gnomAD is not proof of rarity in every population; "
                "gnomAD under-represents some ancestries.",
                "r2.1.1 is the GRCh37 release. A GRCh38 liftover checked against "
                "gnomAD v4 would cover more genomes.",
            ],
        )]

    # --- variant present: report the frequency ------------------------------
    percent, ac, an, detail = combined_frequency(variant)
    if percent is None:
        return [Evidence(
            evidence_id="",
            record_kind="gap",
            candidate_id=cid,
            category="gnomad",
            source="gnomAD (live)",
            record_or_accession=vid,
            query=f"variantId={vid} dataset={DATASET}",
            assembly="GRCh37",
            tool_or_data_version=f"gnomAD GraphQL API ({DATASET})",
            url=url,
            retrieved_at=stamp,
            interpretation=(
                f"gnomAD returned a record for {vid} with no callable alleles "
                "(no exome or genome coverage). Frequency unknown."
            ),
            limitations=["Record present but uncallable; PM2 not assessable."],
        )]

    cand["gnomad_af"] = percent

    if percent >= BA1_PERCENT:
        verdict = (f"At {percent:.4g}% this exceeds the {BA1_PERCENT}% BA1 "
                   "threshold - too common to cause a rare dominant disorder.")
    elif percent >= BS1_PERCENT:
        verdict = (f"At {percent:.4g}% this exceeds the {BS1_PERCENT}% BS1 "
                   "threshold - strong evidence against pathogenicity.")
    else:
        verdict = (f"At {percent:.4g}% this is rare, but PM2 requires absence, "
                   "not rarity; PM2 cannot be claimed.")

    return [Evidence(
        evidence_id="",
        record_kind="retrieved",
        candidate_id=cid,
        category="gnomad",
        source="gnomAD (live)",
        record_or_accession=vid,
        query=f"variantId={vid} dataset={DATASET}",
        assembly="GRCh37",
        raw_field="exome/genome ac | an | af",
        raw_value=f"{ac} | {an} | {percent / 100.0:.6g}",
        tool_or_data_version=f"gnomAD GraphQL API ({DATASET})",
        url=url,
        retrieved_at=stamp,
        interpretation=(
            f"Observed in gnomAD r2.1.1 at {percent:.4g}% "
            f"({ac} alleles of {an}; {detail}). {verdict}"
        ),
        limitations=[
            "Allele frequency is aggregate; a variant rare overall can be "
            "common in a single ancestry group.",
            "gnomAD r2.1.1 excludes individuals with severe pediatric disease "
            "but is not a screened healthy cohort.",
        ],
    )]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live gnomAD frequency lookup (GRCh37).")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", required=True, type=int)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--gene", default="?")
    ap.add_argument("--cache-dir", default="outputs/agent_cache/gnomad")
    args = ap.parse_args(argv)

    cand = {"candidate_id": variant_id(args.chrom, args.pos, args.ref, args.alt),
            "chrom": args.chrom, "pos": args.pos, "ref": args.ref,
            "alt": args.alt, "gene": args.gene}
    for ev in lookup_frequency(cand, cache_dir=args.cache_dir):
        print(f"\n[{ev.source}] {ev.record_or_accession}")
        print(f"  {ev.interpretation}")
        print(f"  {ev.url}")
    print(f"\ncandidate gnomad_af = {cand.get('gnomad_af')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
