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
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence
from ..net import ssl_context

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# esearch answers in JSON; a reader needs the PubMed web search for the same
# query, where the hit list is actually browsable.
PUBMED_SEARCH_UI = "https://pubmed.ncbi.nlm.nih.gov/?term={}"

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


def _cache_name(cache_key: str, query: str) -> str:
    """Cache filename that changes when the query does.

    The key alone described the *intent* of a search ("FGFR2_pathogenic_broad")
    and not the search itself, so editing a query silently reused the previous
    query's answer - a stale count that looks exactly like a fresh one. The
    query digest makes an edited query a cache miss.
    """
    safe = cache_key.replace("/", "_").replace(" ", "_")[:100]
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def esearch(
    query: str,
    db: str = "pubmed",
    retmax: int = 5,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> tuple[int, list[str], str, str, str]:
    """Search NCBI → (count, pmid_list, url, timestamp, query_translation).

    The translation is returned because NCBI silently rewrites queries it
    cannot parse. A caller that needs to know its search terms actually
    survived has to read it - the count alone cannot tell you.
    """
    params = urllib.parse.urlencode({
        "db": db, "term": query, "retmax": retmax,
        "retmode": "json", "sort": "relevance",
    })
    full_url = f"{EUTILS_BASE}/esearch.fcgi?{params}"

    if cache_dir and cache_key:
        safe_key = _cache_name(cache_key, query)
        cache_path = os.path.join(cache_dir, f"esearch_{safe_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return (blob.get("count", 0), blob.get("pmids", []),
                    blob.get("url", full_url), blob.get("retrieved_at", ""),
                    blob.get("translation", ""))

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
        translation = result.get("querytranslation", "") or ""
        stamp = now_iso()

        if cache_dir and cache_key:
            os.makedirs(cache_dir, exist_ok=True)
            safe_key = _cache_name(cache_key, query)
            with open(os.path.join(cache_dir, f"esearch_{safe_key}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"url": full_url, "count": count, "pmids": pmids,
                           "retrieved_at": stamp, "query": query,
                           "translation": translation}, fh, indent=1)

        return count, pmids, full_url, stamp, translation

    except Exception as exc:  # noqa: BLE001
        print(f"  ! PubMed esearch failed: {exc}")
        return 0, [], full_url, "", ""


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


AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}
_HGVS_P = re.compile(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*|=)$")


def protein_change_terms(hgvs_p: str) -> list[str]:
    """Searchable forms of a protein change, or [] if there is nothing to search.

    HGVS wraps a predicted change in parentheses (`p.(Glu563Ala)`) and writes
    `p.?` when the protein effect is unknown. Neither is a search term: NCBI
    parses parentheses as grouping and drops the rest, which silently turns the
    query into the bare letter "p" and returns the whole gene's literature as
    though it were variant-specific. Return the three- and one-letter forms
    only when a real substitution is present.
    """
    core = (hgvs_p or "").strip()
    if core.startswith("p."):
        core = core[2:]
    core = core.strip("()").strip()
    if not core or core in {"?", "="}:
        return []
    m = _HGVS_P.match(core)
    if not m:
        # Unrecognised notation - searching it raw is what caused the bug.
        return []
    ref3, posn, alt3 = m.groups()
    terms = [f"{ref3}{posn}{alt3}"]
    ref1, alt1 = AA3_TO_1.get(ref3), AA3_TO_1.get(alt3)
    if ref1 and alt1:
        # Papers cite either notation; both are the same claim.
        terms.append(f"{ref1}{posn}{alt1}")
    return terms


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
            record_kind="gap",
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
    variant_note = ""
    pc_terms = protein_change_terms(hgvs_p)
    if pc_terms:
        or_clause = " OR ".join(f'"{t}"' for t in pc_terms)
        q = f'"{gene}"[tiab] AND ({or_clause})'
        variant_count, variant_pmids, _, _, xlat = esearch(
            q, retmax=3, cache_dir=cache_dir,
            cache_key=f"{gene}_{pc_terms[0]}_variant")
        # NCBI rewrites what it cannot parse. If none of the terms we asked for
        # survived into the translation, the count belongs to some other query
        # and must not be reported as variant-specific evidence.
        if xlat and not any(t.lower() in xlat.lower() for t in pc_terms):
            variant_note = (f"PubMed did not search the requested variant terms "
                            f"({', '.join(pc_terms)}); its query translation was "
                            f"'{xlat}'. Count discarded.")
            print(f"  ! PubMed query degraded for {gene}: {xlat}")
            variant_count, variant_pmids = 0, []
    elif hgvs_p:
        variant_note = (f"No variant-level search: '{hgvs_p}' is not a "
                        "searchable protein change (unknown or predicted-only "
                        "effect).")

    # --- search 2: gene + disease -------------------------------------------
    gene_disease_count, gene_disease_pmids = 0, []
    if disease:
        disease_clean = disease.split("(")[0].split(",")[0].strip()
        if disease_clean:
            q = f'"{gene}"[tiab] AND "{disease_clean}"'
            gene_disease_count, gene_disease_pmids, _, _, xlat_d = esearch(
                q, retmax=3, cache_dir=cache_dir,
                cache_key=f"{gene}_{disease_clean}_disease")
            # Same guard as the variant search: a disease name NCBI cannot
            # parse would leave a gene-only query returning a count that is
            # not about the disease at all.
            if xlat_d and disease_clean.lower() not in xlat_d.lower():
                print(f"  ! PubMed disease query degraded for {gene}: {xlat_d}")
                gene_disease_count, gene_disease_pmids = 0, []

    # --- search 3: gene + pathogenic (broad) --------------------------------
    q_broad = (f'"{gene}"[tiab] AND (pathogenic[Title/Abstract] OR '
               '"loss of function"[Title/Abstract] OR '
               '"gain of function"[Title/Abstract])')
    gene_path_count, gene_path_pmids, url_broad, stamp, _ = esearch(
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
    elif variant_note:
        parts.append(variant_note)
    elif hgvs_p:
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
        record_kind="retrieved",
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
        url=PUBMED_SEARCH_UI.format(urllib.parse.quote(q_broad)),
        retrieved_at=stamp or now_iso(),
        interpretation=" ".join(parts),
        limitations=([variant_note] if variant_note else []) + [
            "Publication count does not distinguish case reports from "
            "functional studies. A high count means a well-studied gene, "
            "not a proven variant.",
            "Gene terms are matched in title/abstract text, not against a "
            "gene-indexed field; PubMed has no [Gene] tag.",
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
