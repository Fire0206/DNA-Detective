"""PubMed literature search via NCBI E-utilities.

Searches for gene-disease and variant-specific publications to establish
whether a candidate's gene-disease link is supported by published research.
This is a genuinely independent evidence family — PubMed publications are
not derived from ClinVar, VEP, or gnomAD.

Uses the same NCBI E-utilities infrastructure as clinical.py:
    esearch.fcgi  — search PubMed, returns PMIDs and result count
    esummary.fcgi — fetch article title/journal/year for top hits

Rate limit: 3 req/s without API key, 10/s with NCBI_API_KEY.

Usage
-----
    # As an agent tool (registered in main.py):
    from dnadet.tools.pubmed import search_candidate
    rows = search_candidate(candidate_dict)

    # Standalone test:
    python3 -m dnadet.tools.pubmed --gene FGFR2 --disease "Pfeiffer syndrome"
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

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI allows 3 req/s without key. Be conservative.
MIN_INTERVAL = 0.4
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
# E-utilities calls
# --------------------------------------------------------------------------- #


def esearch(
    query: str,
    db: str = "pubmed",
    retmax: int = 5,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> tuple[int, list[str], str, str]:
    """Search NCBI → (count, pmid_list, url, timestamp)."""
    params = urllib.parse.urlencode({
        "db": db, "term": query, "retmax": retmax,
        "retmode": "json", "sort": "relevance",
    })
    full_url = f"{EUTILS_BASE}/esearch.fcgi?{params}"

    if cache_dir and cache_key:
        safe_key = cache_key.replace("/", "_").replace(" ", "_")[:120]
        cache_path = os.path.join(cache_dir, f"esearch_{safe_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return (blob.get("count", 0), blob.get("pmids", []),
                    blob.get("url", full_url), blob.get("retrieved_at", ""))

    _throttle()
    try:
        req = urllib.request.Request(full_url, headers={
            "User-Agent": TOOL_NAME})
        with urllib.request.urlopen(req, timeout=15,
                                    context=ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        result = data.get("esearchresult", {})
        count = int(result.get("count", 0))
        pmids = result.get("idlist", [])[:retmax]
        stamp = now_iso()

        if cache_dir and cache_key:
            os.makedirs(cache_dir, exist_ok=True)
            safe_key = cache_key.replace("/", "_").replace(" ", "_")[:120]
            with open(os.path.join(cache_dir, f"esearch_{safe_key}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"url": full_url, "count": count, "pmids": pmids,
                           "retrieved_at": stamp, "query": query}, fh, indent=1)

        return count, pmids, full_url, stamp

    except Exception as exc:  # noqa: BLE001
        print(f"  ! PubMed esearch failed: {exc}")
        return 0, [], full_url, ""


def esummary(
    pmids: list[str],
    cache_dir: Optional[str] = None,
) -> list[dict[str, str]]:
    """Fetch article summaries for a list of PMIDs."""
    if not pmids:
        return []

    cache_key = "_".join(pmids[:5])
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"esummary_{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)

    _throttle()
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "json",
    })
    try:
        req = urllib.request.Request(
            f"{EUTILS_BASE}/esummary.fcgi?{params}",
            headers={"User-Agent": TOOL_NAME})
        with urllib.request.urlopen(req, timeout=15,
                                    context=ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        results = data.get("result", {})
        articles: list[dict[str, str]] = []
        for pmid in pmids:
            art = results.get(pmid, {})
            if art and isinstance(art, dict):
                authors = art.get("authors") or []
                articles.append({
                    "pmid": pmid,
                    "title": art.get("title", ""),
                    "journal": art.get("source", ""),
                    "pubdate": art.get("pubdate", ""),
                    "first_author": (authors[0].get("name", "")
                                     if authors else ""),
                })

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"esummary_{cache_key}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(articles, fh, indent=1)

        return articles

    except Exception as exc:  # noqa: BLE001
        print(f"  ! PubMed esummary failed: {exc}")
        return []


# --------------------------------------------------------------------------- #
# Per-candidate search
# --------------------------------------------------------------------------- #


def search_candidate(
    cand: dict[str, Any],
    cache_dir: str = "outputs/agent_cache/pubmed",
) -> list[Evidence]:
    """Search PubMed for a candidate's gene-disease and variant evidence.

    Three searches in descending specificity:
    1. Gene + variant HGVS (direct evidence for this mutation)
    2. Gene + candidate disease (gene-disease association)
    3. Gene + pathogenic/mutation (broad — how well-studied is this gene?)

    Returns one Evidence row summarising all three searches.
    """
    gene = cand.get("gene") or ""
    if not gene:
        return [Evidence(
            evidence_id="",
            candidate_id=cand["candidate_id"],
            category="pubmed",
            source="PubMed — skipped",
            interpretation="No gene symbol available; PubMed search requires "
                           "a gene name.",
            limitations=["Cannot search PubMed without a gene name."],
        )]

    cid = cand["candidate_id"]
    hgvs_p = cand.get("hgvs_p") or ""
    disease = cand.get("candidate_disease") or ""

    # --- search 1: gene + variant (most specific) ----------------------------
    variant_count, variant_pmids = 0, []
    if hgvs_p:
        protein_change = hgvs_p.replace("p.", "").strip()
        if protein_change:
            q = f'"{gene}"[Gene] AND ("{protein_change}" OR "{hgvs_p}")'
            variant_count, variant_pmids, _, _ = esearch(
                q, retmax=3, cache_dir=cache_dir,
                cache_key=f"{gene}_{protein_change}_variant")

    # --- search 2: gene + disease -------------------------------------------
    gene_disease_count, gene_disease_pmids = 0, []
    if disease:
        disease_clean = disease.split("(")[0].split(",")[0].strip()
        if disease_clean:
            q = f'"{gene}"[Gene] AND "{disease_clean}"'
            gene_disease_count, gene_disease_pmids, _, _ = esearch(
                q, retmax=3, cache_dir=cache_dir,
                cache_key=f"{gene}_{disease_clean}_disease")

    # --- search 3: gene + pathogenic (broad) --------------------------------
    q_broad = (f'"{gene}"[Gene] AND (pathogenic[Title/Abstract] OR '
               '"loss of function"[Title/Abstract] OR '
               '"gain of function"[Title/Abstract])')
    gene_path_count, gene_path_pmids, url_broad, stamp = esearch(
        q_broad, retmax=3, cache_dir=cache_dir,
        cache_key=f"{gene}_pathogenic_broad")

    # --- fetch summaries for top hits ----------------------------------------
    all_pmids = list(dict.fromkeys(
        variant_pmids + gene_disease_pmids + gene_path_pmids))[:5]
    articles = esummary(all_pmids, cache_dir) if all_pmids else []

    # --- build interpretation ------------------------------------------------
    parts: list[str] = []

    if variant_count > 0:
        parts.append(
            f"PubMed contains {variant_count} publication(s) directly "
            f"referencing {gene} {hgvs_p} — published evidence about this "
            "specific variant.")
    else:
        if hgvs_p:
            parts.append(
                f"No publications directly reference {gene} {hgvs_p}.")

    if gene_disease_count > 0:
        parts.append(
            f"{gene_disease_count} publication(s) link {gene} to "
            f"{disease or 'the candidate disease'}.")
    elif disease:
        parts.append(
            f"No publications found linking {gene} to {disease}.")

    if gene_path_count > 0:
        parts.append(
            f"{gene_path_count} publication(s) describe pathogenic variants "
            f"in {gene} more broadly.")
    else:
        parts.append(
            f"No publications describe pathogenic variants in {gene}.")

    if articles:
        top = "; ".join(
            f"PMID:{a['pmid']} {a['first_author']} et al., {a['journal']} "
            f"({a['pubdate'][:4]})"
            for a in articles[:3])
        parts.append(f"Top results: {top}.")

    if not parts:
        parts.append(f"No relevant publications found for {gene}.")

    return [Evidence(
        evidence_id="",
        candidate_id=cid,
        category="pubmed",
        source="PubMed (NCBI, live)",
        query=f"{gene} + disease/variant",
        raw_field="variant_pubs | gene_disease_pubs | gene_pathogenic_pubs",
        raw_value=f"{variant_count} | {gene_disease_count} | {gene_path_count}",
        tool_or_data_version=(
            f"NCBI E-utilities (eutils.ncbi.nlm.nih.gov), "
            f"retrieved {stamp or now_iso()}"),
        url=url_broad or "",
        retrieved_at=stamp or now_iso(),
        interpretation=" ".join(parts),
        limitations=[
            "Publication count does not distinguish case reports from "
            "functional studies. A high count means a well-studied gene, "
            "not a proven variant.",
            "Search may miss publications indexed under variant synonyms, "
            "alternative gene names, or non-English text.",
        ],
    )]


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PubMed literature search")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--hgvs-p", default="")
    ap.add_argument("--disease", default="")
    ap.add_argument("--cache-dir", default="outputs/pubmed_cache")
    args = ap.parse_args(argv)

    cand: dict[str, Any] = {
        "candidate_id": "cli_test",
        "chrom": "?", "pos": 0, "ref": "?", "alt": "?",
        "gene": args.gene,
        "hgvs_p": args.hgvs_p,
        "candidate_disease": args.disease,
    }
    rows = search_candidate(cand, args.cache_dir)
    for row in rows:
        d = row.to_dict()
        print(f"\n[{d['category']}] {d['source']}")
        print(f"  raw: {d['raw_value']}")
        print(f"  interpretation: {d['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
