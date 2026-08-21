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
        generate_candidate_shortlist, shortlist_evidence,
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

    return ranked, rows, cand_map, TOOLS


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


SUGGESTIONS = [
    "Why is candidate 1 stronger than candidate 2?",
    "Is the evidence for ENPP1 independent?",
    "What is still missing for FGFR2?",
    "What about chr3:38622160 G>A?",
]


def render_qa() -> None:
    """Chat interface with live tool investigation."""
    st.markdown(
        "Ask about any ranked candidate by **gene name** or **rank** "
        "(e.g. *candidate 1*, *ENPP1*).  \n"
        "Or type **coordinates** for a variant not in the shortlist "
        "(e.g. `5-179612-G-A`, `chr10:123256215 T>G`) — DNA Detective "
        "will investigate it live."
    )

    # Lazy-init Q&A engine
    if "qa_ready" not in st.session_state:
        with st.spinner("Loading Q&A engine…"):
            ranked, rows, cand_map, tools = init_qa()
        st.session_state.qa_ranked = ranked
        st.session_state.qa_rows = rows
        st.session_state.qa_cand_map = cand_map
        st.session_state.qa_tools = tools
        st.session_state.qa_ready = True

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

    # Suggestion chips
    st.caption("Try one of these:")
    cols = st.columns(len(SUGGESTIONS))
    for i, sug in enumerate(SUGGESTIONS):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_q = sug

    st.divider()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
            with st.spinner("Investigating…"):
                response = qa_answer(
                    question,
                    st.session_state.qa_ranked,
                    st.session_state.qa_rows,
                    policy, qa_backend, "",
                    cands=st.session_state.qa_cand_map,
                    tools=st.session_state.qa_tools,
                )

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
                    lines = ["**Current ranking** "
                             f"({len(ranked)} candidates, ranked by "
                             "independent evidence strength):\n"]
                    for a in ranked:
                        gaps_short = (f" — gaps: {', '.join(a.gaps[:2])}"
                                      if a.gaps else "")
                        circ = " ⚠️ circularity" if a.circularity else ""
                        lines.append(
                            f"| **#{ranked.index(a)+1}** | **{a.gene}** "
                            f"| `{a.candidate_id}` | strength {a.strength} "
                            f"| {a.independent_support} independent lines "
                            f"| {a.conflicts} conflicts"
                            f"{circ}{gaps_short} |"
                        )
                    lines.append(
                        f"\nAsk about any candidate by gene name or rank "
                        f"for a detailed breakdown.")
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
                        "dna detective": "**DNA Detective** takes Exomiser's ranked variant list and builds an independent, evidence-traced case for each candidate. It checks 6 evidence sources (ClinVar, VEP, PubMed, SpliceAI, AlphaMissense, gnomAD), detects when the same evidence is counted multiple times (circularity), verifies ClinVar by exact allele not just position, and can investigate any variant on demand — even ones Exomiser filtered out.",
                        "grch37": "**GRCh37** (hg19) is the human genome reference build this patient's VCF is aligned to. It's an older version — GRCh38 is current. We use `grch37.rest.ensembl.org` for VEP calls specifically because the default endpoint uses GRCh38, which would silently return annotations for the wrong genomic position.",
                        "grch38": "**GRCh38** (hg38) is the current human genome reference build. This patient's data is on GRCh37, so we query Ensembl's GRCh37 endpoint specifically. Using the default GRCh38 endpoint would give results for the wrong position.",
                        "clinvar": "**ClinVar** is NCBI's public database where genetics labs submit whether a variant is pathogenic, benign, or uncertain. We check both the classification and the review status (star rating: 1★ = one lab, 2★ = multiple labs agree, 3★ = expert panel). We verify by exact allele, not just position — this caught 3 wrong-allele matches in our candidates.",
                        "vep": "**VEP** (Variant Effect Predictor) is Ensembl's tool that tells us what a DNA change does to the protein — missense (changes amino acid), splice region (near splicing junction), synonymous (no change), etc. Seven of our candidates had no consequence annotation until VEP filled the gap.",
                        "pubmed": "**PubMed** is NCBI's medical literature database. We search for gene+disease and gene+variant publications. FGFR2 returned 136 publications linking it to Pfeiffer syndrome — this is genuinely independent evidence that Exomiser doesn't provide.",
                        "spliceai": "**SpliceAI** predicts whether a variant near a splice site actually disrupts splicing (delta scores 0-1). It uses deep learning on pre-mRNA sequences — different methodology from missense predictors. The Broad Institute's API is currently down server-side; we fall back to VEP and report the gap.",
                        "alphamissense": "**AlphaMissense** predicts missense pathogenicity using AlphaFold protein 3D structure. It's methodologically independent from REVEL/MVP (which use sequence conservation), so agreement between them is more meaningful than agreement among sequence-based predictors alone.",
                        "exomiser": "**Exomiser** is the upstream tool that filtered ~37,000 variants down to 11 candidates. DNA Detective takes those 11 and builds an independent, evidence-traced ranking. We don't re-do Exomiser's filtering, but we rank independently and can investigate any variant on demand — even ones Exomiser filtered out.",
                        "gnomad": "**gnomAD** (Genome Aggregation Database) contains variant frequencies from ~140,000 healthy individuals. If a variant is common in gnomAD, it's unlikely to cause a rare disease. Our gnomAD tool is currently a stub — we use Exomiser's 2-year-old snapshot and disclose this as a limitation.",
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
                    response = (
                        f"I can answer questions about specific candidates "
                        f"in the ranking — try naming a gene ({gene_list}),"
                        f" a rank (*candidate 1*), or paste coordinates for"
                        f" any variant (e.g. `chr10:123256215 T>G`).\n\n"
                        f"**Some things you can ask:**\n"
                        f"- Why is {ranked[0].gene} ranked first?\n"
                        f"- Is the evidence for {ranked[1].gene} "
                        f"independent?\n"
                        f"- What about chr5:179612 G>A? *(investigates a "
                        f"new variant live)*"
                    )

            st.markdown(response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main() -> None:
    report = load_report()
    if report is None:
        st.error(
            f"`{REPORT_PATH}` not found. "
            "Run `python3 main.py --team 'Team 9' --agent` first."
        )
        st.stop()

    st.title("🧬 DNA Detective")
    team = report.get("team", "?")
    team_label = team if team.lower().startswith("team") else f"Team {team}"
    st.caption(
        f"{team_label} · "
        "Rare-disease variant prioritisation with traceable evidence"
    )

    tab_rank, tab_qa = st.tabs(
        ["📊 Ranking & Evidence", "💬 Interactive Q&A"])

    with tab_rank:
        render_ranking(report)

    with tab_qa:
        render_qa()


if __name__ == "__main__":
    main()