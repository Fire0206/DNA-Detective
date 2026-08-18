"""Build and write structured submission-compatible reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.models import Candidate, Case, Evidence, FinalReport


def build_report(case: Case, candidates: list[Candidate], evidence: list[Evidence], tools: list[str], team: str = "TEAM_NAME", limitations: list[str] | None = None) -> FinalReport:
    """Assemble a report; callers remain responsible for justified ranking."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return FinalReport(team=team, case=case, tools=tools, run_started_at=now, run_finished_at=now, candidates=candidates, evidence_log=evidence, limitations=limitations or [])


def write_report(report: FinalReport, output_path: Path) -> None:
    """Write formatted JSON to the requested report path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

