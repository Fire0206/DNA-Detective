"""Nocturne theme for the DNA Detective Streamlit app.

One CSS injection plus a handful of HTML builders. Import into app.py:

    from theme import inject_css, hero_card, section_label, timeline_step

Every colour here comes from the Nocturne token set; keep it in step with
.streamlit/config.toml (which themes the widgets Streamlit renders itself).
"""

from __future__ import annotations

import html

# --------------------------------------------------------------------------
# Tokens (mirrors the Nocturne design system)
# --------------------------------------------------------------------------

BG = "#161826"
SURFACE = "#232532"
TEXT = "#e9e9ed"
ACCENT = "#9184d9"

N200, N300, N400, N500, N600, N700, N800, N900 = (
    "#e4e7f5", "#cfd3e5", "#b2b6ca", "#9397ab",
    "#75798c", "#595d6c", "#3f424d", "#292b31",
)
A300, A700, A800, A900 = "#d2cefd", "#5d5294", "#423a6a", "#2b2741"
WARN = "#d9a5a0"

MARKS = {
    "yes": ("\u2713", A300),
    "warn": ("!", WARN),
    "none": ("\u2013", N600),
}


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Inter', system-ui, sans-serif !important;
}

.stApp { background: #161826; color: #e9e9ed; }

/* Streamlit chrome we do not need */
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }

/* Roomier main column, left-aligned per the system */
[data-testid="stAppViewContainer"] > .main .block-container {
  padding: 2.2rem 3rem 4rem; max-width: 1180px;
}

h1, h2, h3, h4 { font-weight: 500 !important; letter-spacing: -0.02em; }
h1 { font-size: 2.4rem !important; }
h2 { font-size: 1.7rem !important; }
p, li, .stMarkdown { color: #cfd3e5; }
a { color: #d2cefd; text-underline-offset: 3px; }
a:hover { color: #9184d9; }
hr { border-color: #292b31; }

/* ── Tabs as a quiet nav bar ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px; border-bottom: 1px solid #292b31; background: transparent;
  margin-bottom: 1.6rem;
}
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 18px; background: transparent;
  border: none; border-bottom: 2px solid transparent;
  color: #9397ab; font-size: 0.94rem;
}
.stTabs [data-baseweb="tab"]:hover { color: #e4e7f5; }
.stTabs [aria-selected="true"] {
  color: #e9e9ed !important; border-bottom-color: #9184d9 !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
  display: none;
}

/* ── Expanders as candidate rows ─────────────────────────────────────── */
[data-testid="stExpander"] {
  border: none !important; border-bottom: 1px solid #1f212d !important;
  border-radius: 0 !important; background: transparent !important;
  box-shadow: none !important; margin: 0 !important;
}
[data-testid="stExpander"] summary {
  padding: 14px 10px !important; border-radius: 8px;
  font-size: 0.95rem; color: #cfd3e5;
}
[data-testid="stExpander"] summary:hover { background: #1c1e2b; }
[data-testid="stExpander"] summary p { color: #cfd3e5 !important; }
[data-testid="stExpander"] summary svg { fill: #75798c; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: 2px 10px 22px 46px; background: transparent;
}

/* ── Buttons: outlined, never filled ────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
  background: transparent; border: 1px solid #3f424d; color: #cfd3e5;
  border-radius: 8px; font-size: 0.85rem; font-weight: 400;
  padding: 0.42rem 0.9rem; transition: border-color .12s, color .12s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #9184d9; color: #d2cefd; background: transparent;
}
.stButton > button:active { background: #2b2741; }
.stButton > button[kind="primary"] {
  border-color: #9184d9; color: #d2cefd; background: transparent;
}
.stButton > button[kind="primary"]:hover { background: #2b2741; }

/* ── File uploaders as drop cards ───────────────────────────────────── */
[data-testid="stFileUploader"] section {
  background: transparent; border: 1px dashed #3f424d; border-radius: 14px;
  padding: 26px 24px; transition: border-color .12s;
}
[data-testid="stFileUploader"] section:hover { border-color: #9184d9; }
[data-testid="stFileUploader"] section small { color: #75798c; }
[data-testid="stFileUploader"] label p {
  font-size: 0.95rem; color: #e9e9ed; font-weight: 500;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
  color: #cfd3e5;
}

/* ── Inputs ─────────────────────────────────────────────────────────── */
/* The chat input is left unstyled so it keeps Streamlit's stock appearance. */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: transparent !important; border: 1px solid #3f424d !important;
  border-radius: 8px !important; color: #e9e9ed !important;
}
.stTextInput input:focus {
  border-color: #9184d9 !important; box-shadow: none !important;
}
/* Chat input excluded: it keeps Streamlit's stock focus styling. */
*:focus-visible:not([data-testid^="stChatInput"]) {
  outline: 2px solid #9184d9 !important; outline-offset: 2px;
}

/* ── Chat ───────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent; padding: 0.3rem 0 1.1rem;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
  background: #2b2741; color: #d2cefd; border: 1px solid #423a6a;
}

/* ── Status / alerts ────────────────────────────────────────────────── */
[data-testid="stStatusWidget"], [data-testid="stExpander"] details { border: none; }
.stAlert { border-radius: 8px; border: 1px solid #3f424d; background: #1c1e2b; }
[data-testid="stMetricValue"] { font-weight: 500; font-size: 1.5rem; }
[data-testid="stMetricLabel"] p {
  font-size: 0.7rem !important; letter-spacing: 0.08em;
  text-transform: uppercase; color: #75798c !important;
}
[data-testid="stSidebar"] {
  background: #191b28; border-right: 1px solid #292b31;
}
code { color: #d2cefd; background: #232532; border-radius: 4px;
       font-size: 0.82em; }
[data-testid="stSpinner"] p { color: #9397ab; }
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTML builders
# --------------------------------------------------------------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def section_label(text: str, note: str = "") -> str:
    note_html = (
        f'<span style="font-size:.78rem;color:{N600};'
        f'margin-left:12px">{_e(note)}</span>' if note else "")
    return (
        f'<div style="font-size:.7rem;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{N600};margin:6px 0 4px">'
        f'{_e(text)}{note_html}</div>')


def brand_header(subtitle: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'margin-bottom:14px">'
        f'<span style="width:26px;height:26px;border-radius:7px;'
        f'border:1px solid {ACCENT};display:inline-flex;align-items:center;'
        f'justify-content:center;font-size:.8rem;color:{A300}">D</span>'
        f'<span style="font-size:.72rem;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{N500}">DNA Detective</span>'
        f'<span style="font-size:.78rem;color:{N600}">{_e(subtitle)}</span>'
        f'</div>')


def hero_card(gene: str, coords: str, verdict: str,
              confidence: str, independent: str, strength: str) -> str:
    """The rank-1 candidate, stated as a conclusion."""
    stats = [
        ("Confidence", confidence, "ordinal, not calibrated"),
        ("Independent lines", independent, "of 6 evidence families"),
        ("Evidence strength", strength, "weighted total"),
    ]
    stat_html = "".join(
        f'<div style="display:flex;flex-direction:column;gap:2px">'
        f'<span style="font-size:.66rem;letter-spacing:.08em;'
        f'text-transform:uppercase;color:{N600}">{_e(label)}</span>'
        f'<span style="font-size:1.35rem;font-weight:500;'
        f'font-variant-numeric:tabular-nums">{_e(value)}</span>'
        f'<span style="font-size:.66rem;color:{N700}">{_e(note)}</span>'
        f'</div>'
        for label, value, note in stats)
    return (
        f'<div style="border:1px solid {A800};border-radius:14px;'
        f'padding:26px 28px;margin:6px 0 26px;'
        f'background:linear-gradient(160deg,#1e1f31,{BG})">'
        f'<div style="display:flex;align-items:center;gap:14px;'
        f'flex-wrap:wrap;margin-bottom:12px">'
        f'<span style="font-size:.66rem;letter-spacing:.12em;color:{A300};'
        f'border:1px solid {A700};border-radius:4px;padding:3px 9px">'
        f'MOST LIKELY</span>'
        f'<span style="font-size:1.85rem;font-weight:500;'
        f'letter-spacing:-.02em;line-height:1">{_e(gene)}</span>'
        f'<span style="font-size:.8rem;font-family:ui-monospace,monospace;'
        f'color:{N500}">{_e(coords)}</span></div>'
        f'<p style="font-size:1rem;line-height:1.55;color:{N200};'
        f'margin:0 0 20px;max-width:78ch;text-wrap:pretty">'
        f'{_e(verdict)}</p>'
        f'<div style="display:flex;gap:40px;flex-wrap:wrap">{stat_html}</div>'
        f'</div>')


def family_line(family: str, mark: str, text: str,
                ids: list[str], circular: bool = False) -> str:
    """One evidence family, in plain English, with its citation IDs."""
    char, colour = MARKS.get(mark, MARKS["none"])
    id_html = "".join(
        f'<span style="font-size:.62rem;font-family:ui-monospace,monospace;'
        f'color:{A300};border:1px solid {A800};border-radius:3px;'
        f'padding:1px 5px;white-space:nowrap">{_e(i)}</span>' for i in ids)
    circ_html = (
        f'<span style="font-size:.62rem;color:{WARN};'
        f'border:1px dashed #7a5450;border-radius:3px;padding:1px 5px;'
        f'white-space:nowrap">not independent</span>' if circular else "")
    body_colour = N600 if mark == "none" else N300
    return (
        f'<div style="display:grid;grid-template-columns:118px 1fr;gap:16px;'
        f'align-items:baseline;padding:5px 0">'
        f'<span style="font-size:.7rem;letter-spacing:.05em;'
        f'text-transform:uppercase;color:{N600}">{_e(family)}</span>'
        f'<div style="display:flex;gap:9px;align-items:baseline;'
        f'flex-wrap:wrap">'
        f'<span style="color:{colour};font-size:.8rem;width:11px;'
        f'flex:none">{char}</span>'
        f'<span style="font-size:.86rem;line-height:1.5;color:{body_colour};'
        f'text-wrap:pretty;flex:1 1 320px">{_e(text)}</span>'
        f'{id_html}{circ_html}</div></div>')


def gap_line(text: str) -> str:
    return (
        f'<div style="display:grid;grid-template-columns:118px 1fr;gap:16px;'
        f'align-items:baseline;padding:9px 0 0;margin-top:8px;'
        f'border-top:1px solid #1f212d">'
        f'<span style="font-size:.7rem;letter-spacing:.05em;'
        f'text-transform:uppercase;color:{N600}">Still missing</span>'
        f'<span style="font-size:.82rem;color:{N500};text-wrap:pretty">'
        f'{_e(text)}</span></div>')


def timeline_step(n: int, title: str, tool: str, body: str,
                  last: bool = False) -> str:
    dot_border = ACCENT if last else N800
    dot_colour = A300 if last else N400
    glyph = "\u25a0" if last else str(n)
    line = ("" if last else
            f'<div style="flex:1;width:1px;min-height:34px;'
            f'background:linear-gradient(#2f3240,#1f212d)"></div>')
    return (
        f'<div style="display:grid;grid-template-columns:56px 1fr;gap:20px;'
        f'align-items:stretch">'
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'gap:6px">'
        f'<span style="width:30px;height:30px;border-radius:99px;'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'font-size:.75rem;flex:none;border:1px solid {dot_border};'
        f'color:{dot_colour}">{glyph}</span>{line}</div>'
        f'<div style="display:flex;flex-direction:column;gap:5px;'
        f'padding-bottom:24px">'
        f'<div style="display:flex;gap:10px;align-items:baseline;'
        f'flex-wrap:wrap">'
        f'<span style="font-size:1rem;font-weight:500;color:{TEXT}">'
        f'{_e(title)}</span>'
        f'<span style="font-size:.66rem;font-family:ui-monospace,monospace;'
        f'color:{N600};border:1px solid {N800};border-radius:3px;'
        f'padding:1px 6px">{_e(tool)}</span></div>'
        f'<span style="font-size:.86rem;line-height:1.55;color:{N400};'
        f'text-wrap:pretty">{_e(body)}</span></div></div>')


def caveat(text: str) -> str:
    return (
        f'<div style="display:flex;gap:8px;align-items:baseline;'
        f'font-size:.78rem;color:#c8968f;margin-top:10px">'
        f'<span>\u25b3</span><span style="text-wrap:pretty">{_e(text)}</span>'
        f'</div>')


def footnote(text: str) -> str:
    return (
        f'<div style="font-size:.72rem;color:{N700};margin-top:28px;'
        f'padding-top:14px;border-top:1px solid {N900}">{_e(text)}</div>')
