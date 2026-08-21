"""DNA Detective — Streamlit frontend.

Displays the pre-generated report (ranking, evidence, limitations) and
provides interactive Q&A with live tool investigation. Ad-hoc variants
not in the Exomiser shortlist can be investigated on demand.

Usage:
    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import re

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DNA Detective", page_icon="🧬", layout="wide")

REPORT_PATH = Path("outputs/dna_detective_report.json")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


@st.cache_resource
def init_qa():
    """Parse the Exomiser JSONL, register live tools, and build assessments.

    This runs ONCE per Streamlit server session. The parse is fast (~50ms),
    and tool results are cached on disk by clinical.py / vep.py, so repeat
    queries do not hit the network.
    """
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

    # Register live tools
    TOOLS["clinvar"] = (
        lambda c: clinvar_process([c], "outputs/agent_cache", False, 1)[0])
    TOOLS["vep"] = lambda c: annotate_candidate(c)
    TOOLS["pubmed"] = lambda c: pubmed_search(c, cache_dir="outputs/agent_cache/pubmed")
    TOOLS["splice"] = lambda c: predict_splice(c, cache_dir="outputs/agent_cache/spliceai")

    from dnadet.tools.alphamissense import lookup_alphamissense
    TOOLS["alphamissense"] = lambda c: lookup_alphamissense(c, cache_dir="outputs/agent_cache/alphamissense")

    # Shortlist + offline evidence (no network calls)
    case = load_case(Path("data/Pfeiffer.vcf"),
                     Path("data/pfeiffer-phenopacket.yml"))
    candidates = generate_candidate_shortlist(case, limit=10)
    evidence = list(shortlist_evidence())

    # Also load agent evidence from the report (if it exists) so the Q&A
    # doesn't re-trigger gap-fill for evidence the agent already gathered.
    report_path = Path("outputs/dna_detective_report.json")
    if report_path.exists():
        import json as _json
        with open(report_path) as _f:
            report_data = _json.load(_f)
        for ev in report_data.get("evidence_log", []):
            # Only add rows not already in offline evidence (agent-produced)
            if ev.get("evidence_id", "").startswith("A"):
                from src.models import Evidence as _Ev
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
    ranked = sorted(
        [assess(c, rows) for c in cand_dicts], key=lambda a: a.sort_key)
    cand_map = {c["candidate_id"]: c for c in cand_dicts}

    # Load drops for the filtered-variants section
    try:
        drops = list(shortlist_drops())
    except Exception:
        drops = []

    return ranked, rows, cand_map, TOOLS, drops


# ---------------------------------------------------------------------------
# Report display helpers
# ---------------------------------------------------------------------------


CATEGORY_ICONS = {
    "clinvar": "🏥", "clingen": "🧪", "vep": "🔬",
    "phenotype": "👤", "frequency": "📊", "exomiser": "📋",
}


def render_ranking(report: dict) -> None:
    """Render the candidate ranking with expandable evidence."""
    candidates = report.get("top_candidates", [])
    all_evidence = report.get("evidence_log", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Candidates ranked", len(candidates))
    col2.metric("Evidence rows", len(all_evidence))
    col3.metric("Tools used", len((report.get("method") or {}).get("tools", [])))

    st.divider()

    # Summary ranking table (quick overview before the detail cards)
    st.markdown("### Variant ranking")
    tbl = ["| Rank | Variant | Gene | Strength | Indep. Lines "
           "| Confidence |",
           "|---|---|---|---|---|---|"]
    for c in candidates:
        reason = c.get("reason_for_rank", "")
        sm = re.match(
            r"evidence strength (\d+) from (\d+) independent", reason)
        strength = sm.group(1) if sm else "?"
        indep = sm.group(2) if sm else "?"
        conf = c.get("confidence", "?")
        if isinstance(conf, float):
            conf = f"{conf:.2f}"
        tbl.append(
            f"| {c.get('rank', '?')} "
            f"| `{c.get('candidate_id', '')}` "
            f"| {c.get('gene', '?')} "
            f"| {strength} | {indep} | {conf} |")
    st.markdown("\n".join(tbl))

    st.divider()

    for c in candidates:
        rank = c.get("rank", "?")
        gene = c.get("gene", "?")
        cid = c.get("candidate_id", "")
        confidence = c.get("confidence", "?")
        reason = c.get("reason_for_rank", "")
        missing = c.get("missing_evidence", [])

        header = f"**#{rank}  {gene}** — `{cid}` · confidence {confidence}"
        with st.expander(header, expanded=(rank == 1)):
            if reason:
                st.markdown(reason)

            cand_ev = [e for e in all_evidence
                       if e.get("candidate_id") == cid]
            if cand_ev:
                for e in cand_ev:
                    eid = e.get("evidence_id", "")
                    cat = e.get("category", "")
                    icon = CATEGORY_ICONS.get(cat, "📎")
                    source = e.get("source", "")
                    interp = e.get("interpretation", "")
                    st.markdown(
                        f"{icon} **[{eid}]** *{source}*\n\n"
                        f"> {interp[:400]}{'…' if len(interp) > 400 else ''}"
                    )

            if missing:
                st.warning(f"**Gaps:** {'; '.join(missing)}")

    # Limitations
    lims = report.get("limitations", [])
    if lims:
        with st.expander("Limitations & disclosures"):
            for lim in lims:
                st.markdown(f"- {lim}")

        # Dropped variants
    if "qa_drops" in st.session_state and st.session_state.qa_drops:
        drops = st.session_state.qa_drops
        with st.expander(
            f"🔍 Filtered variants ({len(drops)} with per-variant reasons)"
        ):
            st.caption(
                "Exomiser filtered ~37,000 variants down to ~280 "
                "candidates. The top 11 became the shortlist above. "
                f"These {len(drops)} variants passed Exomiser's quality "
                "and frequency filters but ranked too low to make the "
                "final cut. You can investigate any of them by pasting "
                "coordinates in the Q&A tab.")
            drop_lines = [
                "| Gene | Variant | Reason |",
                "|---|---|---|",
            ]
            for d in drops:
                if hasattr(d, "to_dict"):
                    d = d.to_dict()
                elif hasattr(d, "__dict__"):
                    d = vars(d)
                elif not isinstance(d, dict):
                    d = {"info": str(d)}
                gene = d.get("gene", d.get("gene_symbol", ""))
                if not gene:
                    reason_text = str(
                        d.get("reason", d.get("drop_reason", "")))
                    gm = re.search(r"gene (\S+)", reason_text)
                    gene = gm.group(1) if gm else "?"
                cid = d.get("candidate_id", d.get("variant_id", "?"))
                reason = str(
                    d.get("reason", d.get("drop_reason",
                           d.get("info", "not specified"))))
                # Escape pipes in reason text
                reason = reason.replace("|", "∣")
                drop_lines.append(
                    f"| {gene} | `{cid}` | {reason} |")
            st.markdown("\n".join(drop_lines))

    # Pre-generated follow-ups
    examples = report.get("follow_up_examples", [])
    if examples:
        with st.expander("Pre-generated follow-up Q&A"):
            for ex in examples:
                st.markdown(f"**Q:** {ex.get('user', '')}")
                st.markdown(ex.get("agent", ""))
                st.divider()


# ---------------------------------------------------------------------------
# Q&A chat
# ---------------------------------------------------------------------------


def _suggestions(ranked: list | None = None) -> list[str]:
    """Dynamic suggestion buttons using gene names from the ranking."""
    if not ranked or len(ranked) < 2:
        return [
            "Show me the ranking",
            "What tools does DNA Detective use?",
            "How does the pipeline work?",
        ]
    top, second = ranked[0].gene, ranked[1].gene
    return [
        f"Why is the {top} variant ranked first?",
        f"Is the {second} variant's evidence independent?",
        f"What evidence is missing for the {top} variant?",
        "Investigate chr3:38622160 G>A",
    ]

# ---------------------------------------------------------------------------
# Agent thinking
# ---------------------------------------------------------------------------


TRANSCRIPT_PATH = Path("docs/transcripts/agent_transcript.md")


def render_agent_thinking() -> None:
    """Display the agent's step-by-step ReAct reasoning process."""
    if not TRANSCRIPT_PATH.exists():
        st.info(
            "No agent transcript found. Run the pipeline with "
            "`--agent` to generate one.")
        return

    content = TRANSCRIPT_PATH.read_text(encoding="utf-8")

    # Split into step blocks at "### Step N"
    parts = re.split(r"(?=### Step \d+)", content)
    header = parts[0].strip() if parts else ""
    steps = [p for p in parts if re.match(r"### Step \d+", p.strip())]

    # --- metrics row ---------------------------------------------------------
    col1, col2 = st.columns(2)
    col1.metric("Agent steps", len(steps))

    # Extract stop reason from the last step
    if steps:
        last = steps[-1]
        stop_match = re.search(
            r"\*\*PRIORITIZE\*\* -> stop.*?\n>\s*(.+?)(?:\n|$)", last,
            re.DOTALL)
        if stop_match:
            reason_text = stop_match.group(1).strip()
            col2.metric(
                "Conclusion",
                reason_text[:70] + "…" if len(reason_text) > 70
                else reason_text)

    st.divider()

    # --- timeline summary (quick-glance table) -------------------------------
    summary_lines = [
        "| Step | Action | Target | Outcome |",
        "|---|---|---|---|",
    ]
    for i, step_text in enumerate(steps, 1):
        is_stop = "**STOP.**" in step_text

        if is_stop:
            summary_lines.append(f"| {i} | 🛑 STOP | — | — |")
            continue

        # Parse the PRIORITIZE line
        act = re.search(
            r"\*\*PRIORITIZE\*\* -> \S+\s+(\S+)\s+(\S+)", step_text)
        if act:
            tool, cid = act.group(1), act.group(2)
            gene_m = re.search(
                rf"\*\*(\w+)\*\*\s*`{re.escape(cid)}`", step_text)
            gene = gene_m.group(1) if gene_m else cid
        else:
            tool, gene = "?", "?"

        # Parse COMPARE outcome (first line after **COMPARE**)
        comp = re.search(
            r"\*\*COMPARE\*\*\s*-\s*\w+:\s*(.+?)(?:\n|$)", step_text)
        outcome = comp.group(1).strip()[:60] if comp else "—"
        if comp and len(comp.group(1).strip()) > 60:
            outcome += "…"

        summary_lines.append(
            f"| {i} | 🔧 {tool} | **{gene}** | {outcome} |")

    st.markdown("\n".join(summary_lines))

    st.divider()

    # --- expandable step details ---------------------------------------------
    st.markdown("##### Step details")
    for i, step_text in enumerate(steps, 1):
        is_stop = "**STOP.**" in step_text

        if is_stop:
            label = f"Step {i}: 🛑 STOP"
        else:
            act = re.search(
                r"\*\*PRIORITIZE\*\* -> \S+\s+(\S+)\s+(\S+)", step_text)
            if act:
                tool, cid = act.group(1), act.group(2)
                gene_m = re.search(
                    rf"\*\*(\w+)\*\*\s*`{re.escape(cid)}`", step_text)
                gene = gene_m.group(1) if gene_m else cid
                label = f"Step {i}: {tool} → {gene}"
            else:
                label = f"Step {i}"

        with st.expander(label, expanded=(i == len(steps))):
            st.markdown(step_text)

    # --- final ranking section (if present) ----------------------------------
    if "## Final ranking" in content:
        final_text = content[content.index("## Final ranking"):]
        with st.expander("📊 Final ranking summary", expanded=False):
            st.markdown(final_text)

def render_qa() -> None:
    """Chat interface with live tool investigation."""
    st.markdown(
        "Ask about any **variant** using its gene name "
        "(e.g. *FGFR2*, *ENPP1*) or coordinates "
        "(e.g. `chr3:38622160 G>A`).  \n"
        "Each ranked entry is a specific variant, not a gene — "
        "the gene name is a shorthand for the variant within it."
    )

    # Lazy-init Q&A engine
    if "qa_ready" not in st.session_state:
        with st.spinner("Loading Q&A engine…"):
            ranked, rows, cand_map, tools, drops = init_qa()
        st.session_state.qa_ranked = ranked
        st.session_state.qa_rows = rows
        st.session_state.qa_cand_map = cand_map
        st.session_state.qa_tools = tools
        st.session_state.qa_ready = True
        st.session_state.qa_drops = drops

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: Q&A settings
    with st.sidebar:
        st.subheader("Q&A settings")
        use_llm = st.toggle("LLM phrasing", value=False,
                            help="Use a language model to phrase answers "
                                 "naturally. Requires GROQ_API_KEY or KIMI_API_KEY.")
        if use_llm:
            qa_backend = st.selectbox("Backend", ["groq", "kimi", "ollama"],
                                      index=0)
        else:
            qa_backend = ""

    # Dynamic suggestion chips
    suggestions = _suggestions(st.session_state.get("qa_ranked"))
    st.caption("Try one of these:")
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_q = sug

    st.divider()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("thinking"):
                for step in msg["thinking"]:
                    st.caption(step)

            content = msg["content"]
            cite_text = ""
            for marker in ("\n\n*\u2713 ", "\n\n*\u26a0\ufe0f "):
                idx = content.rfind(marker)
                if idx >= 0:
                    cite_text = content[idx + 2:].rstrip().rstrip("*")
                    content = content[:idx]
                    break

            st.markdown(content)

            if cite_text:
                cited_ids = sorted(set(
                    re.findall(r"\b([EACGQ]\d{3})\b", content)))
                if cited_ids and "qa_rows" in st.session_state:
                    with st.expander(cite_text, expanded=False):
                        for eid in cited_ids:
                            row = next(
                                (r for r in st.session_state.qa_rows
                                 if r.get("evidence_id") == eid),
                                None)
                            if row:
                                cat = row.get("category", "?")
                                src = row.get("source", "?")
                                interp = (
                                    row.get("interpretation", "")
                                    [:150])
                                if len(row.get(
                                    "interpretation", "")) > 150:
                                    interp += "\u2026"
                                st.caption(
                                    f"**[{eid}]** {cat} \u00b7 {src}: "
                                    f"{interp}")

    # Determine the next question
    question = st.chat_input("Ask DNA Detective…")
    if "pending_q" in st.session_state:
        question = st.session_state.pop("pending_q")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            from dnadet.qa import answer as qa_answer

            policy = "llm" if use_llm else "template"
            tool_log: list[str] = []

            with st.status("🧬 Investigating…", expanded=True) as status:
                def _progress(msg: str) -> None:
                    status.caption(msg)
                    tool_log.append(msg)

                # Pre-check: questions about absent variants should not go
                # through qa_answer, which defaults to the top two.
                q_lower = question.lower()
                skip_qa = False
                if any(w in q_lower for w in (
                    "isn't", "is not", "not in the list",
                    "not in the ranking", "why not",
                    "where is", "missing from", "excluded",
                    "dropped", "filtered", "removed",
                )):
                    skip_qa = True
                    _progress("🔍 Checking filtered variants…")
                    drops = st.session_state.get("qa_drops", [])
                    gene_hit = None
                    for d in drops:
                        dd = (d.to_dict() if hasattr(d, "to_dict")
                              else vars(d) if hasattr(d, "__dict__")
                              else d if isinstance(d, dict) else {})
                        g = dd.get("gene", dd.get("gene_symbol", ""))
                        if g and g.lower() in q_lower:
                            gene_hit = dd
                            break

                    if gene_hit:
                        g = gene_hit.get(
                            "gene", gene_hit.get("gene_symbol", "?"))
                        r = gene_hit.get(
                            "reason", gene_hit.get(
                                "drop_reason", "no reason recorded"))
                        cid = gene_hit.get(
                            "candidate_id",
                            gene_hit.get("variant_id", "?"))
                        response = (
                            f"**{g}** (`{cid}`) was filtered by Exomiser: "
                            f"{r}\n\n"
                            "You can still investigate it — paste its "
                            "coordinates in this chat and DNA Detective "
                            "will run VEP, ClinVar, and PubMed on it live.")
                    else:
                        # Extract a gene name from the question for a
                        # targeted response.
                        words = [w.strip("?.,!:;()") for w in question.split()]
                        words = [w for w in words
                                 if w and w.upper() == w and len(w) >= 3
                                 and w not in ("THE", "NOT", "WHY")]
                        gene_guess = words[0] if words else None
                        if gene_guess:
                            response = (
                                f"**{gene_guess}** is not in the current "
                                "shortlist. Exomiser filtered ~37,000 "
                                "variants down to 11 candidates, and "
                                f"{gene_guess} did not survive that "
                                "process.\n\nIf you have the variant's "
                                "coordinates, paste them here (e.g. "
                                "`chr17:41245466 G>A`) and DNA Detective "
                                "will investigate it live — even if "
                                "Exomiser filtered it out.")
                        elif drops:
                            response = (
                                "Exomiser filtered ~37,000 variants in "
                                f"total. Of these, **{len(drops)}** have "
                                "per-variant drop reasons (they passed "
                                "initial filters but were removed in later "
                                "stages). The rest were removed in bulk "
                                "during earlier filter stages without "
                                "per-variant tracking.\n\n"
                                "Check the **Filtered variants** section "
                                "in the Ranking tab to browse the ones "
                                "with reasons.\n\n"
                                "To investigate any specific variant, "
                                "paste its coordinates here (e.g. "
                                "`chr3:38622160 G>A`) — DNA Detective "
                                "will assess it live.")
                        else:
                            response = (
                                "That variant is not in the current "
                                "shortlist. Exomiser filtered ~37,000 "
                                "variants down to 11.\n\nPaste the "
                                "variant's coordinates to investigate "
                                "it live.")

                if not skip_qa:
                    response = qa_answer(
                        question,
                        st.session_state.qa_ranked,
                        st.session_state.qa_rows,
                        policy, qa_backend, "",
                        cands=st.session_state.qa_cand_map,
                        tools=st.session_state.qa_tools,
                        progress=_progress,
                    )

                if any("🔧" in m for m in tool_log):
                    status.update(
                        label="✅ Investigation complete",
                        state="complete", expanded=True)
                else:
                    status.update(
                        label="💭 Done",
                        state="complete", expanded=True)

            # If the system couldn't resolve anything, check if they're
            # asking about the ranking overview before falling back.
            if response.startswith("I could not tell"):
                ranked = st.session_state.qa_ranked
                q_lower = question.lower()

                # Ranking overview questions
                if any(w in q_lower for w in (
                    "ranking", "ranked", "list", "candidates", "all",
                    "summary", "overview", "results", "show me",
                    "what are", "how many",
                )):
                    lines = [
                        f"**Current ranking** ({len(ranked)} candidates, "
                        "ranked by independent evidence strength):\n",
                        "| Rank | Gene | Strength | Independent | "
                        "Conflicts | Notes |",
                        "|---|---|---|---|---|---|",
                    ]
                    for i, a in enumerate(ranked, 1):
                        notes_parts = []
                        if a.circularity:
                            notes_parts.append("⚠️ circularity")
                        if a.gaps:
                            notes_parts.append(
                                f"gaps: {', '.join(a.gaps[:2])}")
                        notes = "; ".join(notes_parts) or "—"
                        lines.append(
                            f"| {i} | **{a.gene}** | {a.strength} "
                            f"| {a.independent_support} | {a.conflicts} "
                            f"| {notes} |")
                    lines.append(
                        "\nAsk about any variant by gene name or "
                        "coordinates for a detailed breakdown.")
                    response = "\n".join(lines)

                # Methodology / concept questions
                elif any(w in q_lower for w in (
                    "grch37", "grch38", "genome build", "assembly",
                    "clinvar", "vep", "pubmed", "spliceai",
                    "alphamissense", "exomiser", "gnomad",
                    "how does", "how do you", "what does", "what is",
                    "what are", "tell me about", "explain",
                    "methodology", "pipeline", "how it works",
                    "what tools", "dna detective", "about this",
                    "what can you", "help",
                )):
                    concepts = {
                        "dna detective": (
                            "**DNA Detective** analyses genetic variants "
                            "to find which one most likely explains a "
                            "patient's symptoms. It checks multiple "
                            "independent databases (clinical records, "
                            "published research, population data, "
                            "computational predictions) and flags when "
                            "evidence that looks independent actually "
                            "comes from the same source."),
                        "grch37": (
                            "**GRCh37** is the coordinate system used to "
                            "locate positions in this patient's genome — "
                            "like a street address for DNA. It's an older "
                            "but widely-used version (the newer one is "
                            "GRCh38). We specifically use the GRCh37 "
                            "version of our tools because the patient's "
                            "data was mapped to this coordinate system. "
                            "Using the wrong version would look up the "
                            "wrong location."),
                        "grch38": (
                            "**GRCh38** is the newer genome coordinate "
                            "system. This patient's data uses the older "
                            "GRCh37, so all our lookups use GRCh37 "
                            "endpoints to avoid checking the wrong "
                            "location."),
                        "clinvar": (
                            "**ClinVar** is a public database where "
                            "genetics laboratories report whether a "
                            "variant is disease-causing (pathogenic), "
                            "harmless (benign), or uncertain. The star "
                            "rating indicates how many labs agree: "
                            "1★ = one lab's opinion, 2★ = multiple labs "
                            "agree, 3★ = expert panel reviewed. We also "
                            "verify that the exact DNA change matches, "
                            "not just the position — this caught 3 "
                            "wrong matches in our analysis."),
                        "vep": (
                            "**VEP** (Variant Effect Predictor) tells us "
                            "what a DNA change does to the protein it "
                            "codes for. For example: does it change an "
                            "amino acid (missense), disrupt splicing, or "
                            "have no effect (synonymous)? This is needed "
                            "to even begin assessing whether a variant "
                            "could cause disease."),
                        "pubmed": (
                            "**PubMed** is a database of published "
                            "medical research. We search it to see if "
                            "other researchers have studied this gene or "
                            "variant in relation to the patient's "
                            "condition. A gene with hundreds of relevant "
                            "papers is much better understood than one "
                            "with none. Note: we check whether papers "
                            "exist and count them, but do not read the "
                            "full text of each paper."),
                        "spliceai": (
                            "**SpliceAI** predicts whether a variant "
                            "disrupts RNA splicing — the process that "
                            "removes non-coding sections from the "
                            "genetic message before it's read. A splice "
                            "disruption can be just as damaging as "
                            "changing an amino acid. The SpliceAI "
                            "service is currently unavailable (server-"
                            "side issue), so we report this gap rather "
                            "than hiding it."),
                        "alphamissense": (
                            "**AlphaMissense** uses 3D protein structure "
                            "(from AlphaFold) to predict whether an "
                            "amino acid change is harmful. It works "
                            "differently from other predictors that use "
                            "DNA sequence patterns, so when they agree, "
                            "that agreement means more."),
                        "exomiser": (
                            "**Exomiser** is the upstream tool that "
                            "narrowed ~37,000 variants down to a "
                            "manageable shortlist. DNA Detective then "
                            "builds its own independent case for each "
                            "survivor. You can also investigate any "
                            "variant Exomiser filtered out by pasting "
                            "its coordinates in this chat."),
                        "gnomad": (
                            "**gnomAD** is a database of genetic "
                            "variants seen in ~140,000 healthy people. "
                            "If a variant is common in gnomAD, it's "
                            "unlikely to cause a rare disease. Our "
                            "gnomAD data comes from a snapshot that's "
                            "about 2 years old — we disclose this as a "
                            "limitation."),
                    }
                    matched_concepts = [
                        v for k, v in concepts.items()
                        if k in q_lower
                    ]
                    if matched_concepts:
                        response = "\n\n".join(matched_concepts)
                    elif any(w in q_lower for w in (
                        "how does", "how do you", "pipeline",
                        "how it works", "what tools", "methodology",
                    )):
                        response = (
                            "**How DNA Detective works:**\n\n"
                            "1. Exomiser filters ~37,000 variants → 11 "
                            "candidates\n"
                            "2. Our agent checks each candidate against "
                            "6 evidence sources: ClinVar, VEP, PubMed, "
                            "SpliceAI, AlphaMissense, and gnomAD\n"
                            "3. It picks tools selectively — the tool "
                            "that most affects the ranking gets called "
                            "first\n"
                            "4. Each candidate is assessed across 6 "
                            "evidence families with independence checks "
                            "and circularity detection\n"
                            "5. The ranking is based on weighted, "
                            "origin-collapsed evidence strength\n\n"
                            "Ask about a specific candidate for details, "
                            "or type coordinates to investigate any "
                            "variant live."
                        )
                    else:
                        response = matched_concepts[0] if matched_concepts else response

                else:
                    gene_list = ", ".join(
                        a.gene for a in ranked[:5] if a.gene)
                    top = ranked[0].gene if ranked else "?"
                    second = ranked[1].gene if len(ranked) > 1 else "?"
                    response = (
                        f"I can answer questions about variants in the "
                        f"ranking — use a gene name ({gene_list}) or "
                        f"paste coordinates for any variant.\n\n"
                        f"**Some things you can ask:**\n"
                        f"- Why is the {top} variant ranked first?\n"
                        f"- Is the {second} variant's evidence "
                        f"independent?\n"
                        f"- What about chr3:38622160 G>A? *(investigates "
                        f"a variant not in the shortlist)*"
                    )

            # Strip citation verification line and show as expandable
            cite_text = ""
            for marker in ("\n\n*\u2713 ", "\n\n*\u26a0\ufe0f "):
                idx = response.rfind(marker)
                if idx >= 0:
                    cite_text = response[idx + 2:].rstrip().rstrip("*")
                    response = response[:idx]
                    break

            st.markdown(response)

            # Expandable citation details
            if cite_text:
                cited_ids = sorted(set(
                    re.findall(r"\b([EACGQ]\d{3})\b", response)))
                with st.expander(cite_text, expanded=False):
                    rows = st.session_state.qa_rows
                    for eid in cited_ids:
                        row = next(
                            (r for r in rows
                             if r.get("evidence_id") == eid), None)
                        if row:
                            cat = row.get("category", "?")
                            src = row.get("source", "?")
                            interp = (row.get("interpretation") or
                                      "")[:150]
                            if len(row.get("interpretation", "")
                                   ) > 150:
                                interp += "\u2026"
                            st.caption(
                                f"**[{eid}]** {cat} \u00b7 {src}: "
                                f"{interp}")
                        else:
                            st.caption(
                                f"**[{eid}]** (from agent evidence)")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response,
             "thinking": tool_log})
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def _run_uploaded_analysis(vcf_file, pheno_file, team_name) -> None:
    """Save uploaded files, run the pipeline, store results."""
    import tempfile

    # Save uploads to temp files
    tmp_dir = Path(tempfile.mkdtemp())
    vcf_path = tmp_dir / vcf_file.name
    pheno_path = tmp_dir / pheno_file.name
    vcf_path.write_bytes(vcf_file.getvalue())
    pheno_path.write_bytes(pheno_file.getvalue())

    with st.status(
        "🧬 Running DNA Detective…", expanded=True,
    ) as status:
        def _prog(msg: str) -> None:
            status.caption(msg)

        try:
            from main import run_analysis
            results = run_analysis(
                str(vcf_path), str(pheno_path),
                team=team_name, progress=_prog)

            status.update(
                label="✅ Analysis complete",
                state="complete", expanded=True)

        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            status.update(
                label="❌ Analysis failed",
                state="error", expanded=True)
            st.error(f"Pipeline error: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Load the written report as dict
    report_path = Path("outputs/dna_detective_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report_dict = json.load(f)
    else:
        st.error("Report was not written. Check the pipeline output.")
        return

    # Store everything in session state
    st.session_state.report = report_dict
    st.session_state.data_source = f"Uploaded: {vcf_file.name}"
    st.session_state.qa_ranked = results["ranked"]
    st.session_state.qa_rows = results["rows"]
    st.session_state.qa_cand_map = results["cand_map"]
    st.session_state.qa_tools = results["tools"]
    st.session_state.qa_drops = results["drops"]
    st.session_state.qa_ready = True
    st.session_state.chat_history = []  # Reset chat for new data

    # The disk report changed; the cached loader would serve the old one.
    load_report.clear()
    st.rerun()


def main() -> None:
    st.title("🧬 DNA Detective")
    st.caption("Rare-disease variant prioritisation with traceable evidence")

    # ------------------------------------------------------------------
    # Sidebar: file upload + settings
    # ------------------------------------------------------------------
    with st.sidebar:
        st.subheader("📂 Analyse new data")
        vcf_file = st.file_uploader(
            "VCF file", type=["vcf"],
            help="Patient variant call file (GRCh37)")
        pheno_file = st.file_uploader(
            "Phenopacket", type=["yml", "yaml", "json"],
            help="Patient phenotype description")

        can_run = vcf_file is not None and pheno_file is not None
        team_name = st.text_input("Team name", value="Team 9")

        if can_run:
            if st.button("🧬 Run Analysis", type="primary",
                         use_container_width=True):
                _run_uploaded_analysis(
                    vcf_file, pheno_file, team_name)

        st.divider()

        # Show data source
        if "data_source" in st.session_state:
            st.caption(
                f"📊 Data: {st.session_state.data_source}")

    # ------------------------------------------------------------------
    # Load report (from session state or disk)
    # ------------------------------------------------------------------
    report = st.session_state.get("report")

    if report is None:
        st.markdown(
            "### Welcome\n\n"
            "Upload a **VCF** file and **phenopacket** in the sidebar "
            "to begin analysing a patient's genetic variants.")

        # Option to load existing analysis if one exists on disk
        if REPORT_PATH.exists():
            st.divider()
            if st.button("📂 Load previous analysis",
                         help="Load results from the last CLI run"):
                st.session_state.report = load_report()
                st.session_state.data_source = "Previous analysis"
                st.rerun()
        st.stop()

    # Team label
    team = report.get("team", "?")
    team_label = team if team.lower().startswith("team") else f"Team {team}"
    st.caption(f"{team_label}")

    # ------------------------------------------------------------------
    # Three tabs
    # ------------------------------------------------------------------
    tab_rank, tab_agent, tab_qa = st.tabs(
        ["📊 Ranking & Evidence", "🤖 Agent Reasoning",
         "💬 Interactive Q&A"])

    with tab_rank:
        render_ranking(report)

    with tab_agent:
        render_agent_thinking()

    with tab_qa:
        render_qa()


if __name__ == "__main__":
    main()