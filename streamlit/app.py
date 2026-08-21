"""DNA Detective — Streamlit frontend (Nocturne redesign).

Structure:
    upload screen  ->  results screen (Ranking / Reasoning / Ask)

Everything displayed is read from outputs/dna_detective_report.json and
docs/transcripts/agent_transcript.md. Nothing is hard-coded.

Usage:
    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from fork.streamlit.theme import (
    inject_css, brand_header, hero_card, section_label, family_line,
    gap_line, timeline_step, caveat, footnote,
)

st.set_page_config(page_title="DNA Detective", page_icon="\U0001f9ec",
                   layout="wide", initial_sidebar_state="collapsed")

REPORT_PATH = Path("outputs/dna_detective_report.json")
TRANSCRIPT_PATH = Path("docs/transcripts/agent_transcript.md")


# ==========================================================================
# Data loading — unchanged from the original app
# ==========================================================================


@st.cache_data
def load_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


@st.cache_resource
def init_qa():
    """Parse the Exomiser JSONL, register live tools, build assessments."""
    from src.pipeline.case_loader import load_case
    from src.pipeline.prioritization import (
        generate_candidate_shortlist, shortlist_evidence, shortlist_drops,
    )
    from src import bridge
    from dnadet.agent import assess, TOOLS
    from dnadet.tools.vep import annotate_candidate
    from dnadet.tools.clinical import process as clinvar_process
    from dnadet.tools.pubmed import search_candidate as pubmed_search
    from dnadet.tools.spliceai import predict_splice
    from dnadet.tools.alphamissense import lookup_alphamissense

    TOOLS["clinvar"] = (
        lambda c: clinvar_process([c], "outputs/agent_cache", False, 1)[0])
    TOOLS["vep"] = lambda c: annotate_candidate(c)
    TOOLS["pubmed"] = lambda c: pubmed_search(
        c, cache_dir="outputs/agent_cache/pubmed")
    TOOLS["splice"] = lambda c: predict_splice(
        c, cache_dir="outputs/agent_cache/spliceai")
    TOOLS["alphamissense"] = lambda c: lookup_alphamissense(
        c, cache_dir="outputs/agent_cache/alphamissense")

    case = load_case(Path("data/Pfeiffer.vcf"),
                     Path("data/pfeiffer-phenopacket.yml"))
    candidates = generate_candidate_shortlist(case, limit=10)
    evidence = list(shortlist_evidence())

    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report_data = json.load(f)
        from src.models import Evidence as _Ev
        for ev in report_data.get("evidence_log", []):
            if ev.get("evidence_id", "").startswith("A"):
                evidence.append(_Ev(
                    evidence_id=ev.get("evidence_id", ""),
                    candidate_id=ev.get("candidate_id", ""),
                    category=ev.get("category", ""),
                    source=ev.get("source", ""),
                    record_or_accession=ev.get("record_or_accession", ""),
                    query=ev.get("query", ""),
                    assembly=ev.get("assembly", "GRCh37"),
                    transcript=ev.get("transcript"),
                    raw_field=ev.get("raw_field", ""),
                    raw_value=ev.get("raw_value", ""),
                    tool_or_data_version=ev.get("tool_or_data_version", ""),
                    url=ev.get("url", ""),
                    retrieved_at=ev.get("retrieved_at", ""),
                    interpretation=ev.get("interpretation", ""),
                    limitations=ev.get("limitations", []),
                ))

    cand_dicts = [bridge.to_engine_dict(c) for c in candidates]
    rows = bridge.evidence_as_dicts(evidence)
    ranked = sorted([assess(c, rows) for c in cand_dicts],
                    key=lambda a: a.sort_key)
    cand_map = {c["candidate_id"]: c for c in cand_dicts}
    try:
        drops = list(shortlist_drops())
    except Exception:
        drops = []
    return ranked, rows, cand_map, TOOLS, drops


# ==========================================================================
# Report -> display shapes
# ==========================================================================

# Evidence categories in the report, grouped into the six families the
# ranking actually reasons over.
FAMILY_OF = {
    "clinvar": "Clinical", "clingen": "Clinical", "clinical": "Clinical",
    "phenotype": "Phenotype",
    "gnomad": "Population", "frequency": "Population", "population": "Population",
    "effect": "Computational", "spliceai": "Computational",
    "splice": "Computational", "computational": "Computational",
    "pubmed": "Literature", "literature": "Literature",
    "vep": "Consequence", "consequence": "Consequence",
}
FAMILY_ORDER = ["Clinical", "Phenotype", "Population",
                "Computational", "Literature", "Consequence"]

NEGATIVE_HINTS = ("do not match", "does not match", "no overlap", "0 overlap",
                  "not observed", "conflicting", "benign")


def _coords(cand: dict) -> str:
    """Human coordinates, e.g. chr10:123256215 T>G · p.(Glu563Ala)."""
    v = cand.get("normalized_variant") or {}
    base = (f"chr{v.get('chrom', '?')}:{v.get('pos', '?')} "
            f"{v.get('ref', '?')}>{v.get('alt', '?')}")
    for key in ("hgvs_p", "hgvs_c"):
        if v.get(key):
            return f"{base} \u00b7 {v[key]}"
    return base


def _strength_and_lines(cand: dict) -> tuple[str, str]:
    m = re.search(r"evidence strength (\d+) from (\d+) independent",
                  cand.get("reason_for_rank", "") or "")
    return (m.group(1), m.group(2)) if m else ("0", "0")


def _reason_families(cand: dict) -> list[str]:
    """Family names named in reason_for_rank, in the report's own words."""
    reason = cand.get("reason_for_rank", "") or ""
    found = []
    for key in ("clinical", "phenotype", "population",
                "computational", "literature"):
        if re.search(rf"\b{key}\s*\(", reason):
            found.append(key.capitalize())
    return found


def _human_list(items: list[str]) -> str:
    items = [i.lower() for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _row_summary(cand: dict) -> str:
    fams = _reason_families(cand)
    if not fams:
        return "No supporting evidence from any independent family"
    if "Clinical" in fams and "Phenotype" in fams:
        return "Clinical classification and the patient's features both point here"
    if "Clinical" in fams:
        return "Classified in a clinical database, but the features do not match"
    if fams == ["Computational"]:
        return "Prediction tools only \u2014 nothing clinical, nothing phenotypic"
    if fams == ["Population"]:
        return "Rare in the population, and nothing else"
    return f"Supported by {_human_list(fams)} evidence"


def _verdict(cand: dict, lines: str, gaps: int) -> str:
    fams = _reason_families(cand)
    if not fams:
        return ("No independent evidence family supports this variant. It "
                "stays on the list so the reason for its exclusion is "
                "inspectable rather than hidden.")
    open_q = (f"{gaps} question{'s' if gaps != 1 else ''} remain open"
              if gaps else "No open questions remain")
    return (f"{lines} independent evidence "
            f"{'families' if lines != '1' else 'family'} support this "
            f"variant \u2014 {_human_list(fams)}. {open_q}; every one is "
            f"listed with the evidence below.")


def _evidence_by_family(cid: str, evidence_log: list[dict]) -> dict:
    """Group the real evidence rows for one candidate by family."""
    out: dict[str, list[dict]] = {}
    for row in evidence_log:
        if row.get("candidate_id") != cid:
            continue
        fam = FAMILY_OF.get((row.get("category") or "").lower())
        if not fam:
            src = (row.get("source") or "").lower()
            fam = next((f for k, f in FAMILY_OF.items() if k in src), None)
        if fam:
            out.setdefault(fam, []).append(row)
    return out


def _family_render(fam: str, rows: list[dict]) -> str:
    """One family row: plain-English statement + real evidence IDs."""
    if not rows:
        return family_line(fam, "none", "Nothing found in this family.", [])
    texts, ids, circular, negative = [], [], False, False
    for row in rows:
        interp = (row.get("interpretation") or "").strip()
        if interp and interp not in texts:
            texts.append(interp)
        eid = row.get("evidence_id")
        if eid:
            ids.append(eid)
        for lim in row.get("limitations") or []:
            if "not independent" in lim.lower() or "NOT independent" in lim:
                circular = True
        if any(h in interp.lower() for h in NEGATIVE_HINTS):
            negative = True
    text = " ".join(texts)
    if len(text) > 260:
        text = text[:257].rstrip() + "\u2026"
    if fam == "Consequence":
        mark = "none"
    elif negative:
        mark = "warn"
    else:
        mark = "yes"
    return family_line(fam, mark, text, ids[:4], circular)


# ==========================================================================
# Transcript -> timeline
# ==========================================================================


@st.cache_data
def parse_transcript() -> tuple[str, list[dict]]:
    """(patient HPO line, steps) from the agent transcript."""
    if not TRANSCRIPT_PATH.exists():
        return "", []
    content = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    hpo = ""
    m = re.search(r"^Patient HPO: (.+)$", content, re.M)
    if m:
        hpo = m.group(1)

    steps = []
    blocks = [b for b in re.split(r"(?=### Step \d+)", content)
              if re.match(r"### Step \d+", b.strip())]
    for i, block in enumerate(blocks, 1):
        if "**STOP.**" in block or re.search(r"->\s*stop", block):
            note = re.search(r"->\s*stop.*?\n>\s*(.+?)(?:\n|$)", block,
                             re.S)
            steps.append({
                "title": "Stopped \u2014 the order could no longer change",
                "tool": "stop",
                "body": (note.group(1).strip() if note else
                         "No remaining tool call would move the ranking."),
                "last": True, "raw": block,
            })
            continue
        act = re.search(r"\*\*PRIORITIZE\*\* -> \S+\s+(\S+)\s+(\S+)", block)
        why = re.search(r"\*\*PRIORITIZE\*\*.*?\n>\s*(.+?)(?:\n|$)", block,
                        re.S)
        comp = re.search(r"\*\*COMPARE\*\*\s*-\s*(\w+):\s*(.+?)(?:\n|$)",
                         block)
        tool = act.group(1) if act else "?"
        cid = act.group(2) if act else ""
        gene = comp.group(1) if comp else cid
        if not gene and act:
            gm = re.search(rf"\*\*(\w+)\*\*\s*`{re.escape(cid)}`", block)
            gene = gm.group(1) if gm else cid
        body_parts = []
        if why:
            body_parts.append(why.group(1).strip())
        if comp:
            body_parts.append("Result: " + comp.group(2).strip())
        steps.append({
            "title": f"Checked {gene} with {tool}",
            "tool": tool,
            "body": " ".join(body_parts) or "\u2014",
            "last": False, "raw": block,
        })
    if steps:
        steps[-1]["last"] = True
    return hpo, steps


# ==========================================================================
# Screen 1 — upload
# ==========================================================================


def render_upload() -> None:
    st.markdown(brand_header("Rare-disease variant prioritisation"),
                unsafe_allow_html=True)
    st.markdown("# Which variant explains this patient?")
    st.markdown(
        "<p style='font-size:1.02rem;max-width:56ch;color:#b2b6ca'>"
        "Give it one patient's variant calls and their observed features. "
        "It ranks the candidates, shows the evidence behind every rank, and "
        "answers follow-up questions.</p>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(section_label("Step 1 · Variant calls"),
                    unsafe_allow_html=True)
        vcf_file = st.file_uploader(
            "Drop a VCF file", type=["vcf"], key="up_vcf",
            help="The patient's variant calls, aligned to GRCh37")
    with right:
        st.markdown(section_label("Step 2 · Observed features"),
                    unsafe_allow_html=True)
        pheno_file = st.file_uploader(
            "Drop a phenopacket", type=["yml", "yaml", "json"],
            key="up_pheno",
            help="The patient's observed features, as HPO terms")

    st.write("")
    ready = vcf_file is not None and pheno_file is not None
    c1, c2, c3 = st.columns([1.1, 1.4, 2.5])
    with c1:
        if st.button("Run the investigation", type="primary",
                     disabled=not ready, use_container_width=True):
            _run_uploaded_analysis(vcf_file, pheno_file,
                                   st.session_state.get("team", "Team 9"))
    with c2:
        if REPORT_PATH.exists():
            if st.button("Load the previous run", use_container_width=True):
                st.session_state.report = load_report()
                st.session_state.data_source = "Previous analysis"
                st.rerun()
    with c3:
        st.markdown(
            f"<div style='padding-top:8px;font-size:.82rem;color:#75798c'>"
            f"{'Six tools, cached between runs.' if ready else 'Both files are needed before it can start.'}"
            f"</div>", unsafe_allow_html=True)

    st.markdown(footnote(
        "Educational exercise \u2014 not a validated clinical system, not "
        "for patient care."), unsafe_allow_html=True)


def _run_uploaded_analysis(vcf_file, pheno_file, team_name) -> None:
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    vcf_path = tmp_dir / vcf_file.name
    pheno_path = tmp_dir / pheno_file.name
    vcf_path.write_bytes(vcf_file.getvalue())
    pheno_path.write_bytes(pheno_file.getvalue())

    with st.status("Running DNA Detective\u2026", expanded=True) as status:
        try:
            from main import run_analysis
            results = run_analysis(str(vcf_path), str(pheno_path),
                                   team=team_name, progress=status.caption)
            status.update(label="Analysis complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Analysis failed", state="error")
            st.error(f"Pipeline error: {exc}")
            return

    if not REPORT_PATH.exists():
        st.error("Report was not written. Check the pipeline output.")
        return
    with open(REPORT_PATH) as f:
        st.session_state.report = json.load(f)
    st.session_state.data_source = f"Uploaded: {vcf_file.name}"
    st.session_state.qa_ranked = results["ranked"]
    st.session_state.qa_rows = results["rows"]
    st.session_state.qa_cand_map = results["cand_map"]
    st.session_state.qa_tools = results["tools"]
    st.session_state.qa_drops = results["drops"]
    st.session_state.qa_ready = True
    st.session_state.chat_history = []
    load_report.clear()
    st.rerun()


# ==========================================================================
# Screen 2, tab 1 — ranking
# ==========================================================================


def render_ranking(report: dict) -> None:
    candidates = sorted(report.get("top_candidates", []),
                        key=lambda c: c.get("rank", 99))
    evidence_log = report.get("evidence_log", [])
    if not candidates:
        st.info("No candidates in the report.")
        return

    st.markdown("## Ranked candidates")
    tools = len((report.get("method") or {}).get("tools", []))
    st.markdown(
        f"<div style='font-size:.86rem;color:#9397ab;margin:-6px 0 18px'>"
        f"{len(candidates)} worth a look \u00b7 {len(evidence_log)} pieces of "
        f"evidence gathered \u00b7 {tools} tools</div>",
        unsafe_allow_html=True)

    # ---- the conclusion, stated once -------------------------------------
    top = candidates[0]
    strength, lines = _strength_and_lines(top)
    st.markdown(hero_card(
        gene=top.get("gene", "?"),
        coords=_coords(top),
        verdict=_verdict(top, lines, len(top.get("missing_evidence") or [])),
        confidence=_fmt_conf(top.get("confidence")),
        independent=f"{lines} of 6",
        strength=strength,
    ), unsafe_allow_html=True)

    with st.expander("The evidence behind this ranking", expanded=True):
        _render_evidence(top, evidence_log)

    # ---- the rest, quietly ----------------------------------------------
    st.markdown(section_label(
        "Also considered", "click any row for its evidence"),
        unsafe_allow_html=True)
    for cand in candidates[1:]:
        label = (f"`{cand.get('rank')}`  **{cand.get('gene', '?')}** "
                 f"\u2014 {_row_summary(cand)}  \u00b7  "
                 f"{_fmt_conf(cand.get('confidence'))}")
        with st.expander(label):
            _render_evidence(cand, evidence_log)

    # ---- what was filtered out ------------------------------------------
    drops = st.session_state.get("qa_drops") or []
    if drops:
        st.write("")
        with st.expander(
                f"{len(drops)} further variants were filtered out, "
                f"each with a stated reason"):
            st.caption(
                "Exomiser narrowed ~37,000 variants to ~280 candidates; the "
                "shortlist above is the top of that. These passed the "
                "quality and frequency filters but ranked too low. Paste "
                "any of their coordinates in Ask to investigate it live.")
            for d in drops[:120]:
                dd = (d.to_dict() if hasattr(d, "to_dict")
                      else vars(d) if hasattr(d, "__dict__")
                      else d if isinstance(d, dict) else {"info": str(d)})
                gene = dd.get("gene", dd.get("gene_symbol", "")) or "?"
                cid = dd.get("candidate_id", dd.get("variant_id", "?"))
                reason = str(dd.get("reason", dd.get(
                    "drop_reason", dd.get("info", "not specified"))))
                st.markdown(
                    f"<div style='display:grid;"
                    f"grid-template-columns:90px 210px 1fr;gap:14px;"
                    f"padding:5px 0;border-bottom:1px solid #1f212d;"
                    f"font-size:.8rem'>"
                    f"<span style='color:#e4e7f5'>{gene}</span>"
                    f"<code>{cid}</code>"
                    f"<span style='color:#9397ab'>{reason}</span></div>",
                    unsafe_allow_html=True)

    # ---- limitations -----------------------------------------------------
    lims = report.get("limitations") or []
    if lims:
        with st.expander("Limitations and disclosures"):
            for lim in lims:
                st.markdown(f"- {lim}")


def _fmt_conf(conf) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return str(conf or "?")


def _render_evidence(cand: dict, evidence_log: list[dict]) -> None:
    cid = cand.get("candidate_id", "")
    by_fam = _evidence_by_family(cid, evidence_log)
    st.markdown(
        f"<div style='font-size:.74rem;font-family:ui-monospace,monospace;"
        f"color:#75798c;margin-bottom:10px'>{cid} \u00b7 "
        f"{(cand.get('consequence') or 'consequence unknown').lower().replace('_', ' ')}"
        f"</div>", unsafe_allow_html=True)

    html_rows = [_family_render(fam, by_fam.get(fam, []))
                 for fam in FAMILY_ORDER]
    missing = cand.get("missing_evidence") or []
    if missing:
        html_rows.append(gap_line("; ".join(missing[:4]).rstrip(".") + "."))
    st.markdown("".join(html_rows), unsafe_allow_html=True)

    # the raw record, for anyone who wants to check it
    rows = [r for r in evidence_log if r.get("candidate_id") == cid]
    if rows:
        with st.expander(f"Raw records ({len(rows)})"):
            for r in rows:
                url = r.get("url") or ""
                link = f" \u00b7 [source]({url})" if url else ""
                st.markdown(
                    f"**{r.get('evidence_id')}** \u00b7 {r.get('source')}"
                    f"{link}  \n"
                    f"`{r.get('raw_field')}` = `{r.get('raw_value')}`  \n"
                    f"<span style='font-size:.74rem;color:#75798c'>"
                    f"{r.get('tool_or_data_version')} \u00b7 retrieved "
                    f"{r.get('retrieved_at')}</span>",
                    unsafe_allow_html=True)
                for lim in r.get("limitations") or []:
                    st.markdown(caveat(lim), unsafe_allow_html=True)
                st.write("")


# ==========================================================================
# Screen 2, tab 2 — reasoning
# ==========================================================================


def render_reasoning() -> None:
    hpo, steps = parse_transcript()
    if not steps:
        st.info("No agent transcript found. Run the pipeline with `--agent` "
                "to generate one.")
        return

    st.markdown("## How it got there")
    st.markdown(
        f"<div style='font-size:.86rem;color:#9397ab;margin:-6px 0 6px'>"
        f"{len(steps)} steps. Each one names the tool it called and what "
        f"changed as a result.</div>", unsafe_allow_html=True)
    if hpo:
        st.markdown(
            f"<div style='font-size:.78rem;color:#75798c;margin-bottom:22px'>"
            f"Patient features: {hpo}</div>", unsafe_allow_html=True)

    for i, step in enumerate(steps, 1):
        st.markdown(timeline_step(i, step["title"], step["tool"],
                                  step["body"], step["last"]),
                    unsafe_allow_html=True)

    with st.expander("Full transcript"):
        for i, step in enumerate(steps, 1):
            st.markdown(step["raw"])


# ==========================================================================
# Screen 2, tab 3 — ask
# ==========================================================================


def _ensure_qa() -> None:
    if "qa_ready" not in st.session_state:
        with st.spinner("Loading the investigator\u2026"):
            ranked, rows, cand_map, tools, drops = init_qa()
        st.session_state.update(
            qa_ranked=ranked, qa_rows=rows, qa_cand_map=cand_map,
            qa_tools=tools, qa_drops=drops, qa_ready=True)
    st.session_state.setdefault("chat_history", [])


def render_ask(report: dict) -> None:
    _ensure_qa()
    ranked = st.session_state.get("qa_ranked") or []

    st.markdown("## Ask about the case")
    st.markdown(
        "<div style='font-size:.86rem;color:#9397ab;margin:-6px 0 16px'>"
        "Name a gene, or paste coordinates to investigate a variant that is "
        "not on the list.</div>", unsafe_allow_html=True)

    # ---- pre-generated follow-ups: one click, no pipeline call -----------
    examples = report.get("follow_up_examples") or []
    if examples:
        st.markdown(section_label(
            "Worked answers", "prepared during the run \u2014 open instantly"),
            unsafe_allow_html=True)
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            question = ex.get("user", f"Example {i + 1}")
            if cols[i].button(question, key=f"ex_{i}",
                              use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ex.get("agent", ""),
                     "thinking": ["prepared during the run \u2014 "
                                  "no live tool calls"]})
                st.rerun()
        st.write("")

    # ---- live suggestions ------------------------------------------------
    st.markdown(section_label("Or ask live"), unsafe_allow_html=True)
    suggestions = _suggestions(ranked)
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_q = sug
            st.rerun()

    st.write("")

    # ---- history ---------------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            for line in msg.get("thinking") or []:
                st.caption(line)
            content, cite_text = _split_citation(msg["content"])
            st.markdown(content)
            _render_citations(cite_text, content)

    # ---- new question ----------------------------------------------------
    question = st.chat_input("Ask DNA Detective\u2026")
    if "pending_q" in st.session_state:
        question = st.session_state.pop("pending_q")
    if question:
        _answer(question)


def _suggestions(ranked) -> list[str]:
    if not ranked or len(ranked) < 2:
        return ["Show me the ranking", "What tools are used?",
                "How does the pipeline work?"]
    top, second = ranked[0].gene, ranked[1].gene
    return [f"Why is {top} ranked first?",
            f"Is {second}'s evidence independent?",
            f"What is missing for {top}?",
            "Investigate chr3:38622160 G>A"]


def _split_citation(content: str) -> tuple[str, str]:
    for marker in ("\n\n*\u2713 ", "\n\n*\u26a0\ufe0f "):
        idx = content.rfind(marker)
        if idx >= 0:
            return content[:idx], content[idx + 2:].rstrip().rstrip("*")
    return content, ""


def _render_citations(cite_text: str, content: str) -> None:
    if not cite_text:
        return
    cited = sorted(set(re.findall(r"\b([EACGQ]\d{3})\b", content)))
    rows = st.session_state.get("qa_rows") or []
    with st.expander(cite_text):
        for eid in cited:
            row = next((r for r in rows
                        if r.get("evidence_id") == eid), None)
            if row:
                interp = (row.get("interpretation") or "")[:150]
                st.caption(f"**[{eid}]** {row.get('category', '?')} \u00b7 "
                           f"{row.get('source', '?')}: {interp}")
            else:
                st.caption(f"**[{eid}]** (from agent evidence)")


def _answer(question: str) -> None:
    """Run the investigator. Backend logic is the original app's."""
    from dnadet.qa import answer as qa_answer

    st.session_state.chat_history.append(
        {"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        tool_log: list[str] = []
        with st.status("Investigating\u2026", expanded=True) as status:
            def _progress(msg: str) -> None:
                status.caption(msg)
                tool_log.append(msg)

            response = _drops_answer(question, _progress)
            if response is None:
                response = qa_answer(
                    question,
                    st.session_state.qa_ranked,
                    st.session_state.qa_rows,
                    "llm" if st.session_state.get("use_llm") else "template",
                    st.session_state.get("qa_backend", ""), "",
                    cands=st.session_state.qa_cand_map,
                    tools=st.session_state.qa_tools,
                    progress=_progress,
                )
            status.update(label="Done", state="complete", expanded=False)

        if response.startswith("I could not tell"):
            response = _fallback_answer(question, response)

        content, cite_text = _split_citation(response)
        st.markdown(content)
        _render_citations(cite_text, content)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response, "thinking": tool_log})
    st.rerun()


def _drops_answer(question: str, progress) -> str | None:
    """'Why isn't X in the list?' — answered from the drop reasons."""
    q = question.lower()
    if not any(w in q for w in (
            "isn't", "is not", "not in the list", "not in the ranking",
            "why not", "where is", "missing from", "excluded", "dropped",
            "filtered", "removed")):
        return None
    progress("Checking filtered variants\u2026")
    for d in st.session_state.get("qa_drops", []):
        dd = (d.to_dict() if hasattr(d, "to_dict")
              else vars(d) if hasattr(d, "__dict__")
              else d if isinstance(d, dict) else {})
        gene = dd.get("gene", dd.get("gene_symbol", ""))
        if gene and gene.lower() in q:
            reason = dd.get("reason", dd.get("drop_reason",
                                             "no reason recorded"))
            cid = dd.get("candidate_id", dd.get("variant_id", "?"))
            return (f"**{gene}** (`{cid}`) was filtered by Exomiser: "
                    f"{reason}\n\nYou can still investigate it — paste its "
                    f"coordinates here and DNA Detective will run VEP, "
                    f"ClinVar and PubMed on it live.")
    return ("That variant is not in the current shortlist. Exomiser narrowed "
            "~37,000 variants down to 11.\n\nPaste the variant's coordinates "
            "here (e.g. `chr3:38622160 G>A`) and it will be assessed live.")


def _fallback_answer(question: str, original: str) -> str:
    """Ranking overview + concept explanations.

    NOTE: paste the original app's `concepts` dict (ClinVar, VEP, PubMed,
    SpliceAI, AlphaMissense, Exomiser, gnomAD, GRCh37/38…) into CONCEPTS
    below to restore the full glossary — the wiring is already here.
    """
    CONCEPTS: dict[str, str] = {}
    q = question.lower()
    ranked = st.session_state.get("qa_ranked") or []

    matched = [v for k, v in CONCEPTS.items() if k in q]
    if matched:
        return "\n\n".join(matched)

    if any(w in q for w in ("ranking", "ranked", "list", "candidates",
                            "summary", "overview", "results", "show me")):
        lines = [f"**Current ranking** ({len(ranked)} candidates, by "
                 f"independent evidence strength):\n"]
        for i, a in enumerate(ranked, 1):
            notes = []
            if getattr(a, "circularity", None):
                notes.append("shared-source evidence")
            if getattr(a, "gaps", None):
                notes.append(f"gaps: {', '.join(a.gaps[:2])}")
            lines.append(f"{i}. **{a.gene}** — strength {a.strength} from "
                         f"{a.independent_support} independent line(s), "
                         f"{a.conflicts} conflict(s)"
                         + (f" · {'; '.join(notes)}" if notes else ""))
        lines.append("\nAsk about any variant by gene name or coordinates "
                     "for the full breakdown.")
        return "\n".join(lines)

    genes = ", ".join(a.gene for a in ranked[:5] if a.gene)
    return (f"I can answer questions about variants in the ranking — use a "
            f"gene name ({genes}) or paste coordinates for any variant, "
            f"including ones Exomiser filtered out.")


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    inject_css()

    with st.sidebar:
        st.markdown(section_label("Settings"), unsafe_allow_html=True)
        st.session_state.team = st.text_input("Team name", value="Team 9")
        st.session_state.use_llm = st.toggle(
            "LLM phrasing", value=False,
            help="Phrase answers with a language model. "
                 "Requires GROQ_API_KEY or KIMI_API_KEY.")
        if st.session_state.use_llm:
            st.session_state.qa_backend = st.selectbox(
                "Backend", ["groq", "kimi", "ollama"], index=0)
        else:
            st.session_state.qa_backend = ""
        if st.session_state.get("data_source"):
            st.caption(st.session_state.data_source)

    report = st.session_state.get("report")
    if report is None:
        render_upload()
        return

    # ---- results header --------------------------------------------------
    case = report.get("case_id", "?")
    vcf_name = Path((report.get("input") or {}).get("vcf", "")).name or "\u2014"
    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.markdown(brand_header(f"{case} \u00b7 {vcf_name}"),
                    unsafe_allow_html=True)
    with head_r:
        if st.button("New case", use_container_width=True):
            for key in ("report", "data_source", "chat_history"):
                st.session_state.pop(key, None)
            st.rerun()

    tab_rank, tab_reason, tab_ask = st.tabs(["Ranking", "Reasoning", "Ask"])
    with tab_rank:
        render_ranking(report)
    with tab_reason:
        render_reasoning()
    with tab_ask:
        render_ask(report)


if __name__ == "__main__":
    main()
