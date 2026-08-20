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

    # Register live tools
    TOOLS["clinvar"] = (
        lambda c: clinvar_process([c], "outputs/agent_cache", False, 1)[0])
    TOOLS["vep"] = lambda c: annotate_candidate(c)

    # Shortlist + offline evidence (no network calls)
    case = load_case(Path("data/Pfeiffer.vcf"),
                     Path("data/pfeiffer-phenopacket.yml"))
    candidates = generate_candidate_shortlist(case, limit=10)
    evidence = list(shortlist_evidence())

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

            # If the system couldn't resolve anything, give a helpful
            # conversational fallback instead of the terse default.
            if response.startswith("I could not tell"):
                ranked = st.session_state.qa_ranked
                gene_list = ", ".join(a.gene for a in ranked[:5] if a.gene)
                response = (
                    f"I can answer questions about specific candidates in the "
                    f"ranking — try naming a gene ({gene_list}), a rank "
                    f"(*candidate 1*), or paste coordinates for any variant "
                    f"(e.g. `chr10:123256215 T>G`).\n\n"
                    f"**Some things you can ask:**\n"
                    f"- Why is {ranked[0].gene} ranked first?\n"
                    f"- Is the evidence for {ranked[1].gene} independent?\n"
                    f"- What about chr5:179612 G>A? *(investigates a new variant live)*"
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
