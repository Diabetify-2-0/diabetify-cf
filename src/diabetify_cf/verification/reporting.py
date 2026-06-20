from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diabetify_cf.verification.runner import MetricSummary, ScenarioAggregate, ScenarioRunRecord


def build_report_payload(
    *,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        },
        "summary": {
            "immutable_violation_rate": summary.immutable_violation_rate,
            "mutable_violation_rate": summary.mutable_violation_rate,
            "repeatability_rate": summary.repeatability_rate,
            "average_latency_ms": summary.average_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "total_scenarios": summary.total_scenarios,
            "total_runs": summary.total_runs,
            "total_candidates": summary.total_candidates,
        },
        "scenarios": [_scenario_payload(aggregate) for aggregate in aggregates],
    }


def write_report_json(*, path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scenario_payload(aggregate: ScenarioAggregate) -> dict[str, Any]:
    return {
        "name": aggregate.scenario.name,
        "description": aggregate.scenario.description,
        "tags": list(aggregate.scenario.tags),
        "repeat_count": aggregate.scenario.repeat_count,
        "passed": aggregate.passed,
        "repeatability_consistent": aggregate.repeatability_consistent,
        "average_duration_ms": aggregate.average_duration_ms,
        "runs": [_run_payload(run) for run in aggregate.runs],
    }


def _run_payload(run: ScenarioRunRecord) -> dict[str, Any]:
    return {
        "iteration": run.iteration,
        "duration_ms": run.duration_ms,
        "passed": run.passed,
        "expectation_matched": run.expectation_matched,
        "response_status": run.response.status.value,
        "response_reason_code": run.response.reason_code.value,
        "candidate_count": 1 if run.response.candidate is not None else 0,
        "verification": {
            "outcome_consistent": run.verification.outcome_consistent,
            "candidate_count": run.verification.candidate_count,
            "immutable_violation_rate": run.verification.immutable_violation_rate,
            "mutable_violation_rate": run.verification.mutable_violation_rate,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "immutable_ok": item.immutable_ok,
                    "mutable_ok": item.mutable_ok,
                    "prediction_matches_returned": item.prediction_matches_returned,
                    "medical_ok": item.medical_ok,
                    "directional_ok": item.directional_ok,
                    "transition_ok": item.transition_ok,
                }
                for item in run.verification.candidate_results
            ],
        },
    }
