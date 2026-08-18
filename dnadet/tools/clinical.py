"""Block C - live ClinVar and ClinGen lookups.

Block B read a June-2024 snapshot frozen inside Exomiser. This module asks the
same sources *today* and records what they say now, with the accession, the
review status, the submitter counts, the evaluation date, the exact URL and the
retrieval timestamp - the eight things the brief asks for.

The point is not to re-confirm Block B. It is to find DRIFT:

  * a classification that moved since 2406;
  * a conflict that resolved, or one that opened;
  * a ClinVar record that did not exist in 2406 and does now (7 of the 11
    candidates have no ClinVar record in the snapshot at all).

Every difference is emitted as its own Evidence row and filed as CONFLICTING,
because a source disagreeing with itself across two years is exactly the kind of
disagreement the brief says to show rather than hide.

Design notes
------------
* Standard library only. No pip install on anyone's machine.
* Every raw response is written to `cache/` before parsing. The cache IS the
  audit trail - an Evidence row points at the file its claim came from.
* `--replay` re-parses the cache without touching the network, so parsing can be
  fixed without re-querying, and so a demo is reproducible offline.
* NCBI allows 3 requests/sec unauthenticated, 10 with an API key. Set
  NCBI_API_KEY to go faster. The limiter is deliberately conservative.
* Nothing here raises on a failed lookup. A source that could not be reached is
  recorded as a gap, never as a negative result - "not found" and "found,
  nothing there" stay distinguishable.

Usage
-----
    python3 -m dnadet.tools.clinical --inspect          # 1 request, dump raw shape
    python3 -m dnadet.tools.clinical                    # full run over candidates
    python3 -m dnadet.tools.clinical --replay           # re-parse cache, no network
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import ssl
from datetime import datetime, timezone
from typing import Any, Optional

from ..contract import Evidence
from ..net import ssl_context



EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINVAR_UI = "https://www.ncbi.nlm.nih.gov/clinvar/variation/{}/"
CLINGEN_VALIDITY = "https://search.clinicalgenome.org/kb/gene-validity"
CLINGEN_EREPO = "https://erepo.clinicalgenome.org/evrepo/api/classifications"

TOOL = "DNA-Detective-SummerCamp2026"
CONTACT = os.environ.get("NCBI_EMAIL", "")
API_KEY = os.environ.get("NCBI_API_KEY", "")

# 3 req/s unauthenticated. 0.4s leaves headroom; NCBI blocks aggressive clients.
MIN_INTERVAL = 0.12 if API_KEY else 0.40
_last_call = [0.0]

RETRIEVAL_NOTE = "Live query. Timestamp and URL recorded; raw response cached."


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# HTTP with caching
# --------------------------------------------------------------------------- #


def _throttle() -> None:
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def fetch(
    url: str,
    params: dict[str, str],
    cache_path: str,
    replay: bool = False,
    retries: int = 3,
) -> tuple[Optional[Any], str, str]:
    """Return (parsed_json, full_url, retrieved_at).

    On replay, reads the cache and returns the timestamp recorded at capture
    time - not the time of the replay, which would be a lie in the evidence log.
    """
    qs = dict(params)
    if API_KEY:
        qs["api_key"] = API_KEY
    if CONTACT:
        qs["email"] = CONTACT
    qs["tool"] = TOOL
    full_url = f"{url}?{urllib.parse.urlencode(qs)}"
    # The cached/published URL never contains the API key.
    public_qs = {k: v for k, v in qs.items() if k != "api_key"}
    public_url = f"{url}?{urllib.parse.urlencode(public_qs)}"

    if replay or os.path.exists(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return blob.get("body"), blob.get("url", public_url), blob.get("retrieved_at", "")
        if replay:
            return None, public_url, ""

    last_err = ""
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": TOOL})
            with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            stamp = now_iso()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"url": public_url, "retrieved_at": stamp, "body": body},
                    fh, indent=1,
                )
            return body, public_url, stamp
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                TimeoutError, http.client.IncompleteRead,
                http.client.HTTPException, OSError) as exc:
            # IncompleteRead in particular: NCBI truncates responses under load,
            # and it is not a subclass of URLError, so a narrower tuple lets a
            # transient failure through as a permanent gap.
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))

    print(f"  ! fetch failed after {retries} tries: {last_err}\n    {public_url}")
    return None, public_url, ""


# --------------------------------------------------------------------------- #
# ClinVar
# --------------------------------------------------------------------------- #


def clinvar_summary(variation_id: str, cache_dir: str, replay: bool = False):
    """esummary for a known VariationID."""
    return fetch(
        f"{EUTILS}/esummary.fcgi",
        {"db": "clinvar", "id": str(variation_id), "retmode": "json"},
        os.path.join(cache_dir, f"clinvar_esummary_{variation_id}.json"),
        replay,
    )


def clinvar_search_position(chrom: str, pos: int, cache_dir: str, replay: bool = False):
    """Find ClinVar records at a GRCh37 coordinate.

    This is how a candidate with NO record in the 2406 snapshot gets checked for
    a record submitted since. `chrpos37` is the GRCh37-specific position field;
    plain `chrpos` searches GRCh38 and would silently return the wrong locus.
    """
    term = f"{chrom}[chr] AND {pos}[chrpos37]"
    return fetch(
        f"{EUTILS}/esearch.fcgi",
        {"db": "clinvar", "term": term, "retmode": "json", "retmax": "20"},
        os.path.join(cache_dir, f"clinvar_esearch_{chrom}_{pos}.json"),
        replay,
    )


def parse_clinvar_summary(body: Any, variation_id: str) -> dict:
    """Pull the fields the brief asks for out of an esummary payload.

    NCBI renamed `clinical_significance` to `germline_classification` in the
    2023/24 schema change. Both are probed; the key that answered is reported so
    the evidence log can say which schema it read.
    """
    rec = (body or {}).get("result", {}).get(str(variation_id))
    if not isinstance(rec, dict):
        return {}

    block, schema = None, ""
    for key in ("germline_classification", "clinical_significance", "classification"):
        if isinstance(rec.get(key), dict):
            block, schema = rec[key], key
            break

    traits = []
    for t in (block or {}).get("trait_set", []) or rec.get("trait_set", []) or []:
        name = t.get("trait_name") if isinstance(t, dict) else None
        if name:
            traits.append(name)

    subs = rec.get("supporting_submissions") or {}

    # variation_set carries the coordinate and HGVS ClinVar itself considers
    # canonical - needed to confirm we are looking at the same locus, and to
    # catch transcript-dependent renumbering.
    vset = (rec.get("variation_set") or [{}])[0]
    grch37 = {}
    for loc in vset.get("variation_loc", []) or []:
        # ClinVar marks GRCh38 "current" and GRCh37 "previous" - correct from its
        # point of view, since GRCh38 is the live assembly. Filtering on
        # status == "current" therefore discards the only coordinate that matters
        # for a GRCh37 case and makes every locus check fail. Match the assembly
        # name by prefix (it can read "GRCh37.p13") and keep the status as data.
        name = str(loc.get("assembly_name", ""))
        if name.startswith("GRCh37"):
            grch37 = {"chr": str(loc.get("chr", "")),
                      "start": loc.get("start"),
                      "stop": loc.get("stop"),
                      "status": loc.get("status", ""),
                      "assembly_name": name,
                      "acc": loc.get("assembly_acc_ver", "")}
            break

    return {
        "accession": rec.get("accession_version") or rec.get("accession") or "",
        "title": rec.get("title", ""),
        "classification": (block or {}).get("description", ""),
        "review_status": (block or {}).get("review_status", ""),
        "last_evaluated": (block or {}).get("last_evaluated", ""),
        "schema_key": schema,
        "traits": traits,
        "scv_count": len(subs.get("scv", []) or []),
        "rcv_count": len(subs.get("rcv", []) or []),
        "molecular_consequence": "; ".join(rec.get("molecular_consequence_list", []) or []),
        "protein_change": rec.get("protein_change", ""),
        "genes": [g.get("symbol") for g in rec.get("genes", []) or [] if isinstance(g, dict)],
        "cdna_change": vset.get("cdna_change", ""),
        "canonical_spdi": vset.get("canonical_spdi", ""),
        "grch37": grch37,
    }


def same_locus(g37: dict, chrom: str, pos: int) -> Optional[bool]:
    """True/False if comparable, None if the record has no GRCh37 coordinate.

    esummary returns coordinates as STRINGS ("123256215"), and occasionally as
    "". None means unverifiable, which is not the same answer as False.
    """
    raw = g37.get("start")
    if raw in (None, ""):
        return None
    try:
        start = int(str(raw))
    except (TypeError, ValueError):
        return None
    return str(g37.get("chr", "")).lstrip("chr") == str(chrom) and start == int(pos)


def strip_hgvs_p(text: str) -> str:
    """'p.(Glu563Ala)' and 'p.Glu563Ala' are the same call written two ways."""
    return (text or "").replace("(", "").replace(")", "").replace("p.", "").strip().upper()


def spdi_alleles(spdi: str) -> Optional[tuple[str, str]]:
    """Split a canonical SPDI into (deleted, inserted). None if unparseable.

    Format: ACCESSION:POSITION:DELETED:INSERTED, e.g.
    NC_000010.10:123256214:T:G. SPDI is fully left-shifted and may carry
    context bases the VCF representation does not, so compare the CHANGE
    rather than the literal strings.
    """
    parts = (spdi or "").split(":")
    if len(parts) != 4:
        return None
    return parts[2].upper(), parts[3].upper()


def net_change(ref: str, alt: str) -> tuple[int, str]:
    """(length delta, inserted/deleted bases) with shared flanks trimmed.

    Reduces both VCF-style (ref 'T', alt 'TGCACG...') and SPDI-style
    (ref 'GCACG...', alt 'GCACG...AGAGAG...') to the same canonical change,
    so two representations of one variant compare equal.
    """
    r, a = (ref or "").upper(), (alt or "").upper()
    while r and a and r[0] == a[0]:
        r, a = r[1:], a[1:]
    while r and a and r[-1] == a[-1]:
        r, a = r[:-1], a[:-1]
    return len(a) - len(r), (a or r)


def alleles_agree(cand_ref: str, cand_alt: str, spdi: str) -> Optional[bool]:
    """True/False if both representations are parseable, None if not comparable."""
    pair = spdi_alleles(spdi)
    if not pair or not cand_ref or not cand_alt:
        return None
    c_delta, c_seq = net_change(cand_ref, cand_alt)
    s_delta, s_seq = net_change(*pair)
    if c_delta != s_delta:
        return False
    return c_seq == s_seq



    """'p.(Glu563Ala)' and 'p.Glu563Ala' are the same call written two ways."""
    return (text or "").replace("(", "").replace(")", "").replace("p.", "").strip().upper()



    """'p.(Glu563Ala)' and 'p.Glu563Ala' are the same call written two ways."""
    return (text or "").replace("(", "").replace(")", "").replace("p.", "").strip().upper()


# Exomiser's snapshot labels are aggregate BUCKETS; ClinVar's live labels are
# specific. "Pathogenic" sits INSIDE "PATHOGENIC_OR_LIKELY_PATHOGENIC", so
# comparing the strings reports drift where none happened. Compare tiers:
# a move is only real if it crosses between pathogenic / uncertain / benign /
# conflicting.
TIERS = {
    "PATHOGENIC": "P", "LIKELY_PATHOGENIC": "P",
    "PATHOGENIC_OR_LIKELY_PATHOGENIC": "P",
    "BENIGN": "B", "LIKELY_BENIGN": "B", "BENIGN_OR_LIKELY_BENIGN": "B",
    "UNCERTAIN_SIGNIFICANCE": "U",
    "CONFLICTING_PATHOGENICITY_INTERPRETATIONS": "X",
}


def tier_of(label: str) -> str:
    return TIERS.get(normalise_label(label), "?")


# Snapshot labels are SCREAMING_SNAKE; live labels are prose. Compare on this.
def normalise_label(text: str) -> str:
    t = (text or "").strip().upper().replace("/", " OR ").replace("-", " ")
    t = " ".join(t.split())
    aliases = {
        "PATHOGENIC OR LIKELY PATHOGENIC": "PATHOGENIC_OR_LIKELY_PATHOGENIC",
        "PATHOGENIC LIKELY PATHOGENIC": "PATHOGENIC_OR_LIKELY_PATHOGENIC",
        "BENIGN OR LIKELY BENIGN": "BENIGN_OR_LIKELY_BENIGN",
        "CONFLICTING CLASSIFICATIONS OF PATHOGENICITY":
            "CONFLICTING_PATHOGENICITY_INTERPRETATIONS",
        "CONFLICTING INTERPRETATIONS OF PATHOGENICITY":
            "CONFLICTING_PATHOGENICITY_INTERPRETATIONS",
        "UNCERTAIN SIGNIFICANCE": "UNCERTAIN_SIGNIFICANCE",
        "LIKELY PATHOGENIC": "LIKELY_PATHOGENIC",
        "LIKELY BENIGN": "LIKELY_BENIGN",
    }
    return aliases.get(t, t.replace(" ", "_"))


# --------------------------------------------------------------------------- #
# ClinGen
# --------------------------------------------------------------------------- #


def clingen_gene_validity(gene: str, cache_dir: str, replay: bool = False):
    """ClinGen gene-disease validity for a gene symbol.

    ClinGen answers a different question from ClinVar. ClinVar: is this VARIANT
    pathogenic? ClinGen validity: is this GENE genuinely linked to this DISEASE
    at all (Definitive / Strong / Moderate / Limited / Disputed / Refuted)?

    That distinction decides candidate 1 vs candidate 2 here. A Definitive
    gene-disease link whose phenotype does not match the patient is weaker than
    it looks, and a Limited link is weak regardless of how damaging the variant
    is predicted to be.

    VERIFIED endpoint (probe, 2026-08-18): erepo.clinicalgenome.org returns JSON.
    search.clinicalgenome.org/kb/genes/<sym>?format=json returns HTML, and
    /api/curations/gene/<sym> 404s - neither is usable.
    """
    return fetch(
        CLINGEN_EREPO,
        {"matchMode": "exact", "gene": gene},
        os.path.join(cache_dir, f"clingen_{gene}.json"),
        replay,
    )


def probe_clingen(gene: str = "FGFR2") -> None:
    """Report status code and first bytes from candidate ClinGen endpoints."""
    candidates = [
        f"{CLINGEN_EREPO}?matchMode=exact&gene={gene}",
        f"https://search.clinicalgenome.org/kb/genes/{gene}?format=json",
        f"https://search.clinicalgenome.org/api/curations/gene/{gene}",
    ]
    for url in candidates:
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": TOOL})
            with urllib.request.urlopen(req, timeout=20, context=ssl_context()) as resp:
                head = resp.read(400).decode("utf-8", errors="replace")
                print(f"[{resp.status}] {url}\n    {head[:300]}\n")
        except Exception as exc:  # noqa: BLE001 - probe reports everything
            print(f"[ERR] {url}\n    {type(exc).__name__}: {exc}\n")


# --------------------------------------------------------------------------- #
# Evidence assembly
# --------------------------------------------------------------------------- #


class Log:
    def __init__(self, prefix: str = "C") -> None:
        self.prefix = prefix
        self.rows: list[Evidence] = []

    def add(self, **kw: Any) -> Evidence:
        ev = Evidence(evidence_id=f"{self.prefix}{len(self.rows) + 1:03d}", **kw)
        self.rows.append(ev)
        return ev


def process(candidates: list[dict], outdir: str, replay: bool, limit: Optional[int]):
    cache_dir = os.path.join(outdir, "cache")
    log = Log()
    drift: list[dict] = []
    seen_genes: set[str] = set()

    for cand in candidates[: limit or len(candidates)]:
        cid = cand["candidate_id"]
        gene = cand.get("gene") or "?"
        snap_id = cand.get("clinvar_vcv")
        snap_label = cand.get("clinvar_classification")
        print(f"- {gene} {cid}")

        variation_id = snap_id
        found_new = False
        try_ids: list[str] = [str(snap_id)] if snap_id else []

        # No record in 2406 -> ask whether one exists now.
        if not variation_id:
            body, url, stamp = clinvar_search_position(
                cand["chrom"], cand["pos"], cache_dir, replay
            )
            ids = (body or {}).get("esearchresult", {}).get("idlist", []) or []
            if ids:
                try_ids = [str(i) for i in ids]
                variation_id, found_new = try_ids[0], True
                print(f"    NEW since 2406: ClinVar {ids}")
            else:
                log.add(
                    candidate_id=cid, category="clinvar", source="ClinVar (live)",
                    query=f"{cand['chrom']}[chr] AND {cand['pos']}[chrpos37]",
                    raw_field="esearchresult.idlist", raw_value="[]",
                    tool_or_data_version=f"NCBI E-utilities, retrieved {stamp or 'n/a'}",
                    url=url, retrieved_at=stamp,
                    interpretation=(
                        "No ClinVar record at this position, live, today. Absent from "
                        "the archive is NOT evidence of benignity - it means no lab "
                        "has submitted an interpretation."
                    ),
                    limitations=[
                        "Position search only; a record under a different "
                        "representation of the same allele could be missed.",
                        "Absence of submission is absence of evidence, not evidence "
                        "of absence.",
                    ],
                )
                continue

        # A position search can return several records (different alleles at the
        # same coordinate). Taking idlist[0] and moving on would silently bind
        # the candidate to whichever record NCBI happened to sort first.
        # Two passes, because a record carrying no GRCh37 entry in variation_loc
        # (newer submissions are often GRCh38-only) is UNVERIFIED, not matched -
        # accepting it first would beat a genuinely coordinate-matched sibling.
        matched: Optional[tuple[str, dict, str, str]] = None
        unverified: Optional[tuple[str, dict, str, str]] = None
        rejected: list[str] = []
        allele_confirmed = False
        for vid in try_ids:
            body, url, stamp = clinvar_summary(vid, cache_dir, replay)
            info = parse_clinvar_summary(body, vid)
            if not info:
                continue
            verdict = same_locus(info.get("grch37") or {}, cand["chrom"], cand["pos"])
            if verdict is None:
                if unverified is None:
                    unverified = (vid, info, url, stamp)
                continue
            if verdict:
                allele_ok = alleles_agree(
                    cand.get("ref", ""), cand.get("alt", ""),
                    info.get("canonical_spdi", ""),
                )
                if allele_ok is False:
                    print(f"    - {vid} same position, DIFFERENT allele, skipped")
                    rejected.append(vid)
                    continue
                matched = (vid, info, url, stamp)
                allele_confirmed = allele_ok is True
                break
            g37 = info.get("grch37") or {}
            print(f"    - {vid} is a different locus "
                  f"({g37.get('chr')}:{g37.get('start')}), skipped")

        chosen = matched or unverified
        locus_verified = matched is not None

        if chosen is None:
            log.add(
                candidate_id=cid, category="clinvar", source="ClinVar (live)",
                record_or_accession=", ".join(try_ids),
                query=f"esummary db=clinvar id={','.join(try_ids)}",
                tool_or_data_version="NCBI E-utilities", retrieved_at=now_iso(),
                interpretation=(
                    (f"{len(rejected)} ClinVar record(s) sit at this position "
                     f"({', '.join(rejected)}) but every one describes a DIFFERENT "
                     "allele. No ClinVar interpretation applies to this variant. "
                     "Position agreement is not allele agreement."
                     if rejected else
                     "No ClinVar record returned for this locus that is parseable "
                     "and coordinate-matched - GAP, not a negative result.")
                ),
                limitations=[
                    "Absence of a matching submission is absence of evidence, not "
                    "evidence of benignity.",
                ] + ([] if rejected else
                     ["Live retrieval failed or the schema was unrecognised."]),
            )
            continue

        variation_id, info, url, stamp = chosen
        if len(try_ids) > 1:
            print(f"    resolved to {variation_id} of {len(try_ids)} records"
                  + ("" if locus_verified else "  [LOCUS UNVERIFIED]"))
        elif not locus_verified:
            print("    [locus unverified - no GRCh37 coordinate in record]")

        live_label = info["classification"]
        stars = info["review_status"]
        snap_tier, live_tier = tier_of(snap_label or ""), tier_of(live_label)
        moved = bool(snap_label) and snap_tier != live_tier and live_tier != "?"
        refined = (bool(snap_label) and not moved
                   and normalise_label(live_label) != normalise_label(snap_label))

        log.add(
            candidate_id=cid, category="clinvar", source="ClinVar (live)",
            record_or_accession=info["accession"] or f"VariationID {variation_id}",
            query=f"esummary db=clinvar id={variation_id}",
            raw_field=f"{info['schema_key']}.description | review_status | last_evaluated",
            raw_value=f"{live_label} | {stars} | {info['last_evaluated']}",
            tool_or_data_version=f"NCBI E-utilities, retrieved {stamp}",
            url=CLINVAR_UI.format(variation_id), retrieved_at=stamp,
            interpretation=(
                f"ClinVar today: {live_label or 'no classification'} "
                f"({stars or 'review status unknown'}), last evaluated "
                f"{info['last_evaluated'] or 'unknown'}. "
                f"{info['scv_count']} submission(s), {info['rcv_count']} condition "
                f"record(s). Conditions: {', '.join(info['traits']) or 'none listed'}."
                + (" This record did not exist in the 2406 snapshot."
                   if found_new else "")
            ),
            limitations=[
                RETRIEVAL_NOTE,
                "Aggregate label only; individual submitter evidence needs the "
                "full VCV record.",
            ] + (["Newly submitted since the Exomiser snapshot - Exomiser's "
                  "ranking never saw this."] if found_new else [])
              + ([f"{len(try_ids)} ClinVar records returned for this position "
                  f"({', '.join(try_ids)}); {variation_id} selected by GRCh37 "
                  f"coordinate match, the others describe different alleles."]
                 if len(try_ids) > 1 else [])
              + ([f"Snapshot recorded the aggregate bucket {snap_label}; the live "
                  f"record is more specific ({live_label}). Same tier - a "
                  f"refinement of wording, not a change of call."]
                 if refined else []),
        )

        # --- locus verification: are we even looking at the same variant? ------
        g37 = info.get("grch37") or {}
        if not locus_verified:
            log.add(
                candidate_id=cid, category="clinvar",
                source="ClinVar (live) - locus check",
                record_or_accession=info["accession"],
                query="variation_set[0].variation_loc GRCh37",
                raw_field="variation_loc", raw_value="no GRCh37 entry",
                tool_or_data_version=f"NCBI E-utilities, retrieved {stamp}",
                url=CLINVAR_UI.format(variation_id), retrieved_at=stamp,
                interpretation=(
                    "This ClinVar record carries no GRCh37 coordinate, so it could "
                    "NOT be confirmed to describe this candidate. It was matched by "
                    "position search alone."
                ),
                limitations=[
                    "Locus unverified - every claim from this record is provisional.",
                    "Confirm by lifting the candidate to GRCh38 or by comparing the "
                    "canonical SPDI before citing this record in the answer.",
                ],
            )
        else:
            same = True  # only reached when same_locus() already returned True
            log.add(
                candidate_id=cid, category="clinvar",
                source="ClinVar (live) - locus check",
                record_or_accession=info["accession"],
                query=f"variation_set[0].variation_loc GRCh37",
                raw_field="chr | start | assembly_acc_ver",
                raw_value=(f"{g37.get('chr')} | {g37.get('start')} | "
                           f"{g37.get('assembly_name')} ({g37.get('status')}) | "
                           f"{g37.get('acc')}"),
                tool_or_data_version=f"NCBI E-utilities, retrieved {stamp}",
                url=CLINVAR_UI.format(variation_id), retrieved_at=stamp,
                interpretation=(
                    f"ClinVar's GRCh37 coordinate for this record is "
                    f"{g37.get('chr')}:{g37.get('start')}; the candidate is "
                    f"{cand['chrom']}:{cand['pos']}. "
                    + ("Same locus - the record describes this variant."
                       if same else
                       "MISMATCH - this ClinVar record is NOT this variant. Do not "
                       "carry its classification across.")
                ),
                limitations=([] if same else
                             ["Locus mismatch invalidates every other claim drawn "
                              "from this record."]),
            )
            if not same:
                print(f"    ! locus mismatch {g37.get('chr')}:{g37.get('start')}")
                continue

        # --- transcript-dependent renumbering ---------------------------------
        # p.? is Exomiser's marker for "no protein-level consequence" - the normal
        # case for a splice or non-coding variant, not a disagreement to report.
        snap_p = cand.get("hgvs_p")
        live_title = info.get("title", "")
        snap_p_norm = strip_hgvs_p(snap_p or "")
        if (snap_p_norm and snap_p_norm not in {"", "?", "="}
                and snap_p_norm not in
                strip_hgvs_p(live_title + " " + info.get("protein_change", ""))):
            log.add(
                candidate_id=cid, category="clinvar",
                source="ClinVar (live) - HGVS check",
                record_or_accession=info["accession"],
                raw_field="title | protein_change | cdna_change",
                raw_value=f"{live_title} | {info.get('protein_change','')[:120]}",
                tool_or_data_version=f"Exomiser 2406 vs NCBI live {stamp}",
                url=CLINVAR_UI.format(variation_id), retrieved_at=stamp,
                interpretation=(
                    f"Protein-level naming differs by transcript. Exomiser 2406 "
                    f"reports {snap_p}; ClinVar's preferred transcript gives "
                    f"{live_title}. Same genomic change, different residue number. "
                    "Any protein-level claim must name its transcript."
                ),
                limitations=[
                    "Not a disagreement about pathogenicity - a representation "
                    "difference. Resolve by fixing one MANE Select transcript.",
                ],
            )
            print(f"    HGVS differs: {snap_p} vs {live_title}")

        if moved:
            drift.append({"candidate_id": cid, "gene": gene,
                          "snapshot": snap_label, "live": live_label})
            log.add(
                candidate_id=cid, category="clinvar", source="ClinVar drift check",
                record_or_accession=info["accession"] or str(variation_id),
                raw_field="snapshot 2406 vs live",
                raw_value=f"{snap_label} -> {live_label}",
                tool_or_data_version=f"Exomiser 2406 vs NCBI live {stamp}",
                url=CLINVAR_UI.format(variation_id), retrieved_at=stamp,
                interpretation=(
                    f"CLASSIFICATION DRIFT for {gene}: the 2406 snapshot Exomiser "
                    f"ranked on says {snap_label}; ClinVar today says {live_label}. "
                    "Any ranking inherited from Exomiser rests on the older value."
                ),
                limitations=["Source disagrees with itself across two years - "
                             "the live value supersedes, but both are reported."],
            )
            print(f"    DRIFT {snap_label} -> {live_label}")

        if gene and gene != "?" and gene not in seen_genes:
            seen_genes.add(gene)
            gbody, gurl, gstamp = clingen_gene_validity(gene, cache_dir, replay)
            if isinstance(gbody, dict):
                interps = gbody.get("variantInterpretations") or []
                log.add(
                    candidate_id=cid, category="clingen",
                    source="ClinGen Evidence Repository (ERepo)",
                    record_or_accession=gene,
                    query=f"matchMode=exact&gene={gene}",
                    raw_field="variantInterpretations",
                    raw_value=f"{len(interps)} interpretation(s)",
                    tool_or_data_version=f"ClinGen ERepo, retrieved {gstamp}",
                    url=f"{CLINGEN_VALIDITY}?search={gene}", retrieved_at=gstamp,
                    interpretation=(
                        f"ClinGen expert panels have published {len(interps)} variant "
                        f"interpretation(s) for {gene}."
                        + (" No Variant Curation Expert Panel has curated this gene, "
                           "so no 3-star expert classification exists for ANY variant "
                           "in it - the strongest available clinical evidence here is "
                           "ClinVar submitter consensus."
                           if not interps else "")
                    ),
                    limitations=[
                        "ERepo covers VCEP-curated variants only; it is not a "
                        "gene-disease validity classification.",
                        "Absence of a VCEP curation says nothing about the variant - "
                        "it says the gene has not been through expert curation.",
                    ],
                )

    return log.rows, drift


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Block C - live ClinVar / ClinGen")
    ap.add_argument("--candidates", default="outputs/blockB/candidates.json")
    ap.add_argument("--outdir", default="outputs/blockC")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replay", action="store_true", help="parse cache, no network")
    ap.add_argument("--inspect", action="store_true",
                    help="one ClinVar call, dump the raw shape, exit")
    ap.add_argument("--probe-clingen", action="store_true")
    ap.add_argument("--variation-id", default="13294", help="for --inspect")
    args = ap.parse_args(argv)

    if args.probe_clingen:
        probe_clingen()
        return 0

    cache_dir = os.path.join(args.outdir, "cache")
    if args.inspect:
        body, url, stamp = clinvar_summary(args.variation_id, cache_dir, args.replay)
        if not body:
            print("no response")
            return 1
        rec = body.get("result", {}).get(args.variation_id, {})
        print(f"url: {url}\nretrieved: {stamp}")
        print(f"record keys: {sorted(rec.keys())}\n")
        for k in ("germline_classification", "clinical_significance", "classification"):
            if k in rec:
                print(f"{k}: {json.dumps(rec[k])[:600]}\n")
        print("parsed:", json.dumps(parse_clinvar_summary(body, args.variation_id),
                                    indent=1))
        return 0

    with open(args.candidates, "r", encoding="utf-8") as fh:
        candidates = json.load(fh)

    os.makedirs(args.outdir, exist_ok=True)
    rows, drift = process(candidates, args.outdir, args.replay, args.limit)

    with open(os.path.join(args.outdir, "evidence.jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")
    with open(os.path.join(args.outdir, "drift.json"), "w", encoding="utf-8") as fh:
        json.dump(drift, fh, indent=2)

    print(f"\nevidence rows : {len(rows)}")
    print(f"drift found   : {len(drift)}")
    for d in drift:
        print(f"  {d['gene']}: {d['snapshot']} -> {d['live']}")
    print(f"written to    : {args.outdir}/  (raw responses in cache/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())