# DNA Detective — Nocturne UI

Drop-in replacement for the Streamlit frontend. Two files plus a theme config:

```
app.py                    # replaces your existing app.py
theme.py                  # new — CSS + HTML builders
.streamlit/config.toml    # new — themes the widgets Streamlit renders itself
```

## Installing

1. Copy `theme.py` next to your `app.py`.
2. Copy `.streamlit/config.toml` into the repo root (create the folder if needed).
3. Replace `app.py`, or copy it in as `app_new.py` and run `streamlit run app_new.py` to compare side by side.

No new dependencies. Paths (`outputs/dna_detective_report.json`,
`docs/transcripts/agent_transcript.md`, `data/Pfeiffer.vcf`) are unchanged.

## What changed

**Structure.** Upload is now the landing screen, not a sidebar afterthought:
two drop zones side by side, "Run the investigation" enabled only when both
are in, plus "Load the previous run" when a report exists on disk. The three
tabs (Ranking / Reasoning / Ask) only appear once there is a report.

**Ranking.** The rank-1 candidate is a single conclusion card — gene,
coordinates, a plain-English verdict, and three numbers — with its evidence
open by default. Ranks 2–n are quiet rows that expand. The markdown summary
table is gone; evidence is one row per family (Clinical, Phenotype,
Population, Computational, Literature, Consequence) built by grouping the
real `evidence_log` rows by `category`, showing each family's
`interpretation` text with its actual evidence IDs. Rows whose limitations
say "not independent" get an explicit flag. `Raw records` nests underneath
for anyone checking the field, value, tool version and limitations.

**Reasoning.** The transcript is parsed into a vertical timeline — one step
per `### Step N`, titled from the real `PRIORITIZE` tool + gene, with the
rationale and `COMPARE` outcome as the body. The full markdown is still
available under `Full transcript`.

**Ask.** `follow_up_examples` from the report are restored as one-click
"Worked answers" buttons — clicking one drops the prepared question and its
answer straight into the conversation with no pipeline call, which makes them
reliable in a demo. Live suggestions sit below, then the normal chat input.

## One thing to finish

`_fallback_answer()` has an empty `CONCEPTS: dict[str, str] = {}`. Paste your
existing `concepts` dictionary (ClinVar, VEP, PubMed, SpliceAI,
AlphaMissense, Exomiser, gnomAD, GRCh37/38) into it — the matching and
rendering are already wired. Everything else in the Q&A path
(`dnadet.qa.answer`, the drop-reason pre-check, the citation expander) is
your original logic, unchanged.
