from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diabetify_cf.schemas import JSONFeatureValue
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
            "lof_violation_rate": summary.lof_violation_rate,
            "average_lof_score": summary.average_lof_score,
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


def build_actionability_report_payload(
    *,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
) -> dict[str, Any]:
    return {
        "summary": {
            "immutable_violation_rate": summary.immutable_violation_rate,
            "mutable_violation_rate": summary.mutable_violation_rate,
            "total_scenarios": summary.total_scenarios,
            "total_candidates": summary.total_candidates,
            "average_latency_ms": summary.average_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
        },
        "scenarios": [_actionability_scenario_payload(aggregate) for aggregate in aggregates],
    }


def build_plausibility_report_payload(
    *,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
) -> dict[str, Any]:
    return {
        "summary": {
            "lof_within_threshold": summary.lof_violation_rate in (None, 0.0),
            "average_lof_score": summary.average_lof_score,
            "total_scenarios": summary.total_scenarios,
            "total_candidates": summary.total_candidates,
            "average_latency_ms": summary.average_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
        },
        "scenarios": [_plausibility_scenario_payload(aggregate) for aggregate in aggregates],
    }


def build_repeatability_report_payload(
    *,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
) -> dict[str, Any]:
    return {
        "summary": {
            "repeatability_rate": summary.repeatability_rate,
            "fully_repeatable": summary.repeatability_rate in (None, 1.0),
            "total_scenarios": summary.total_scenarios,
            "total_runs": summary.total_runs,
            "average_latency_ms": summary.average_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
        },
        "scenarios": [_repeatability_scenario_payload(aggregate) for aggregate in aggregates],
    }

def build_latency_report_payload(
    *,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
    target_average_latency_ms: float = 5000.0,
) -> dict[str, Any]:
    all_runs = [run for aggregate in aggregates for run in aggregate.runs]
    durations = [run.duration_ms for run in all_runs]
    successful_runs = sum(1 for run in all_runs if run.passed)
    return {
        "summary": {
            "target_average_latency_ms": target_average_latency_ms,
            "passed": summary.average_latency_ms < target_average_latency_ms
            and successful_runs == len(all_runs),
            "average_latency_ms": summary.average_latency_ms,
            "min_latency_ms": min(durations) if durations else 0,
            "max_latency_ms": max(durations) if durations else 0,
            "p95_latency_ms": summary.p95_latency_ms,
            "total_scenarios": summary.total_scenarios,
            "total_runs": summary.total_runs,
            "successful_runs": successful_runs,
            "failed_runs": len(all_runs) - successful_runs,
        },
        "scenarios": [_latency_scenario_payload(aggregate) for aggregate in aggregates],
    }

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
                    "external_lof_score": item.external_lof_score,
                    "external_lof_within_threshold": item.external_lof_within_threshold,
                    "medical_ok": item.medical_ok,
                    "directional_ok": item.directional_ok,
                    "transition_ok": item.transition_ok,
                }
                for item in run.verification.candidate_results
            ],
        },
    }


def _actionability_scenario_payload(aggregate: ScenarioAggregate) -> dict[str, Any]:
    run = aggregate.runs[0]
    instance_profile = dict(aggregate.scenario.request.instance.features)
    candidate_profile: dict[str, JSONFeatureValue] | None = None
    changed_features: dict[str, dict[str, JSONFeatureValue]] | None = None
    if run.response.candidate is not None:
        candidate_profile = _ordered_profile(
            source=run.response.candidate.features,
            preferred_order=list(instance_profile.keys()),
        )
        changed_features = _changed_features(
            baseline=instance_profile,
            candidate=candidate_profile,
        )

    return {
        "name": aggregate.scenario.name,
        "description": _mutable_description(aggregate.scenario.request.constraints.mutable_allowed),
        "immutable_violation": run.verification.immutable_violation_rate not in (None, 0.0),
        "mutable_violation": run.verification.mutable_violation_rate not in (None, 0.0),
        "instance_profile": instance_profile,
        "counterfactual_candidate": candidate_profile,
        "changed_features": changed_features,
    }


def _plausibility_scenario_payload(aggregate: ScenarioAggregate) -> dict[str, Any]:
    run = aggregate.runs[0]
    lof_score = None
    if run.verification.candidate_results:
        lof_score = run.verification.candidate_results[0].external_lof_score
    return {
        "name": aggregate.scenario.name,
        "description": _mutable_description(aggregate.scenario.request.constraints.mutable_allowed),
        "lof_score": lof_score,
    }


def _repeatability_scenario_payload(aggregate: ScenarioAggregate) -> dict[str, Any]:
    instance_profile = dict(aggregate.scenario.request.instance.features)
    payload: dict[str, Any] = {
        "name": aggregate.scenario.name,
        "description": _mutable_description(aggregate.scenario.request.constraints.mutable_allowed),
        "repeat_count": aggregate.scenario.repeat_count,
        "repeatable": aggregate.repeatability_consistent,
        "all_statuses_identical": _all_statuses_identical(aggregate.runs),
        "all_candidates_identical": _all_candidates_identical(aggregate.runs),
    }
    for run in aggregate.runs:
        candidate_profile = None
        if run.response.candidate is not None:
            candidate_profile = _ordered_profile(
                source=run.response.candidate.features,
                preferred_order=list(instance_profile.keys()),
            )
        payload[f"counterfactual_profile_run_{run.iteration}"] = candidate_profile
    return payload

def _latency_scenario_payload(aggregate: ScenarioAggregate) -> dict[str, Any]:
    durations = [run.duration_ms for run in aggregate.runs]
    statuses: dict[str, int] = {}
    for run in aggregate.runs:
        status = run.response.status.value
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "name": aggregate.scenario.name,
        "description": aggregate.scenario.description,
        "target_probability": aggregate.scenario.request.target.min_target_probability,
        "description_mutable": _mutable_description(
            aggregate.scenario.request.constraints.mutable_allowed
        ),
        "repeat_count": aggregate.scenario.repeat_count,
        "passed": aggregate.passed,
        "average_latency_ms": aggregate.average_duration_ms,
        "min_latency_ms": min(durations) if durations else 0,
        "max_latency_ms": max(durations) if durations else 0,
        "p95_latency_ms": _percentile(durations, 95.0),
        "terminal_statuses": statuses,
        "runs": [
            {
                "iteration": run.iteration,
                "duration_ms": run.duration_ms,
                "passed": run.passed,
                "response_status": run.response.status.value,
                "response_reason_code": run.response.reason_code.value,
            }
            for run in aggregate.runs
        ],
    }

def _mutable_description(mutable_allowed: list[str]) -> str:
    if not mutable_allowed:
        return "Mutable: none"
    labels = ", ".join(_feature_label(name) for name in mutable_allowed)
    return f"Mutable: {labels}"


def _feature_label(name: str) -> str:
    aliases = {
        "BMI": "BMI",
        "is_hypertension": "Hypertension",
        "is_cholesterol": "Cholesterol",
        "is_bloodline": "Bloodline",
        "is_macrosomic_baby": "Macrosomic Baby",
        "moderate_physical_activity_frequency": "Physical Activity",
        "smoking_status": "Smoking Status",
        "brinkman_index": "Brinkman Index",
        "age": "Age",
    }
    if name in aliases:
        return aliases[name]
    normalized = name.removeprefix("is_").replace("_", " ")
    return normalized.title()


def _changed_features(
    *,
    baseline: dict[str, JSONFeatureValue],
    candidate: dict[str, JSONFeatureValue],
) -> dict[str, dict[str, JSONFeatureValue]]:
    changed: dict[str, dict[str, JSONFeatureValue]] = {}
    for feature_name, before_value in baseline.items():
        if feature_name not in candidate:
            continue
        after_value = candidate[feature_name]
        if _equal(before_value, after_value):
            continue
        changed[feature_name] = {
            "before": before_value,
            "after": after_value,
        }
    return changed


def _ordered_profile(
    *,
    source: dict[str, JSONFeatureValue],
    preferred_order: list[str],
) -> dict[str, JSONFeatureValue]:
    ordered: dict[str, JSONFeatureValue] = {}
    for feature_name in preferred_order:
        if feature_name in source:
            ordered[feature_name] = source[feature_name]
    for feature_name, value in source.items():
        if feature_name not in ordered:
            ordered[feature_name] = value
    return ordered


def _equal(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def _all_statuses_identical(runs: list[ScenarioRunRecord]) -> bool:
    if len(runs) <= 1:
        return True
    baseline = runs[0].response.status
    return all(run.response.status == baseline for run in runs[1:])


def _all_candidates_identical(runs: list[ScenarioRunRecord]) -> bool:
    if len(runs) <= 1:
        return True
    baseline = _candidate_signature(runs[0])
    return all(_candidate_signature(run) == baseline for run in runs[1:])


def _candidate_signature(run: ScenarioRunRecord) -> object:
    candidate = run.response.candidate
    if candidate is None:
        return None
    return tuple(
        (feature_name, _normalized_value(value))
        for feature_name, value in sorted(candidate.features.items())
    )


def _normalized_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 9)
    return value
def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


