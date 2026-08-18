# Development guide

This repository is a first-day skeleton for a DNA variant investigation agent. It accepts a VCF and phenotype information, produces a candidate shortlist, can collect optional external evidence, and writes a traceable JSON report. It is not a clinical system and intentionally contains no biological ranking rules or live database integration.

## Layout

- `src/pipeline/case_loader.py` prepares portable `Case` metadata from the VCF and phenotype file.
- `src/pipeline/prioritization.py` is where filtering, annotation handoff, and phenotype-aware candidate ranking belong.
- `src/tools/` contains one file per optional external evidence source. Add a module-level `TOOL = Tool(name, description, investigate)`; tools are discovered automatically, and `investigate(candidate, case)` returns `list[Evidence]`.
- `src/agent/investigator.py` coordinates selected tools and is the place for evidence comparison and reasoning policy.
- `src/models/schemas.py` defines the shared `Candidate`, `Evidence`, and `FinalReport` structures. Their serialized field names align with `starter/submission_template.json`.
- `src/pipeline/reporting.py` creates and writes submission-compatible JSON.

## Run the skeleton

From the repository root, validate the supplied inputs and create an empty, explicitly non-analytic report:

```bash
python main.py --dry-run
```

The report is written to `outputs/dna_detective_report.json`. The normal command (`python main.py`) deliberately stops at `generate_candidate_shortlist` until a collaborator supplies the prioritization implementation.

## Collaboration contract

Do not fabricate evidence, rankings, or biological interpretations. A tool must return traceable `Evidence` records with source, query/accession, assembly, version or retrieval time, URL when available, and limitations. Preserve filter and ranking reasons when implementing `generate_candidate_shortlist`; keep the final report compatible with the starter submission template.

