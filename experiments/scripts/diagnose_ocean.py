from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from diabetify_cf.config import Settings
from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from experiments.scripts.audit_comparison import _latest_comparison_root
from experiments.scripts.collect_results import collect_results
from experiments.scripts.run_benchmark import DEFAULT_OUTPUT_ROOT
from experiments.scripts.run_comparison import engine_from_source_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def _optional_number(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return _number(float(value))
    except (TypeError, ValueError):
        return str(value)


def _parse_json_counter(value: str) -> dict[str, int]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): int(raw_value) for key, raw_value in parsed.items()}


def _scenario_rows_by_engine(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        engine = engine_from_source_file(row.get("source_file", ""))
        scenario = row.get("scenario") or "unknown"
        grouped[engine][scenario] = row
    return {engine: dict(scenarios) for engine, scenarios in sorted(grouped.items())}


def _candidate_rows_by_engine_scenario(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        engine = row.get("engine_name") or engine_from_source_file(row.get("source_file", ""))
        scenario = _scenario_from_candidate_source(row.get("source_file", ""))
        grouped[engine][scenario].append(row)
    return {
        engine: {scenario: list(items) for scenario, items in scenarios.items()}
        for engine, scenarios in sorted(grouped.items())
    }


def _rows_by_engine_scenario_from_source(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        engine = row.get("engine_name") or engine_from_source_file(row.get("source_file", ""))
        scenario = _scenario_from_candidate_source(row.get("source_file", ""))
        grouped[engine][scenario].append(row)
    return {
        engine: {scenario: list(items) for scenario, items in scenarios.items()}
        for engine, scenarios in sorted(grouped.items())
    }


def _scenario_from_candidate_source(source_file: str) -> str:
    parts = Path(source_file).parts
    if "scenarios" not in parts:
        return "unknown"
    index = parts.index("scenarios")
    if index + 1 >= len(parts):
        return "unknown"
    return parts[index + 1]


def _top_changed_features(candidate_rows: list[dict[str, str]], top_n: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in candidate_rows:
        try:
            delta = json.loads(row.get("delta", "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(delta, dict):
            continue
        for feature_name in delta:
            counts[str(feature_name)] += 1
    return [
        {"feature": feature_name, "count": count}
        for feature_name, count in counts.most_common(top_n)
    ]


def _load_cases_for_scenario(row: dict[str, str]) -> list[dict[str, Any]]:
    run_dir = row.get("run_dir")
    if not run_dir:
        return []
    return _read_jsonl(Path(run_dir) / "cases.jsonl")


def _load_inputs_for_scenario(row: dict[str, str]) -> list[dict[str, str]]:
    run_dir = row.get("run_dir")
    if not run_dir:
        return []
    return _read_csv(Path(run_dir) / "inputs.csv")


def _scenario_config_path(row: dict[str, str]) -> Path | None:
    source_file = row.get("source_file")
    scenario = row.get("scenario")
    if not source_file or not scenario:
        return None
    return Path(source_file).parent / scenario / "effective_config.json"


def _expected_outcome(row: dict[str, str]) -> dict[str, Any]:
    config_path = _scenario_config_path(row)
    if config_path is None:
        return {}
    raw = _read_json(config_path).get("expected_outcome") or {}
    return raw if isinstance(raw, dict) else {}


def _is_expected_infeasible(
    *,
    feasible_rate: float,
    reason_counts: dict[str, int],
    expected_outcome: dict[str, Any],
) -> bool:
    if feasible_rate > 0.0:
        return False

    observed_reasons = {reason for reason, count in reason_counts.items() if count > 0}
    if not expected_outcome and observed_reasons == {"NO_MUTABLE_FEATURE"}:
        return True

    if expected_outcome.get("feasible") is not False:
        return False

    expected_reasons = {
        str(reason) for reason in expected_outcome.get("reason_codes", []) if reason
    }
    if not expected_reasons:
        return True
    return bool(observed_reasons) and observed_reasons.issubset(expected_reasons)


def _effective_config(row: dict[str, str]) -> dict[str, Any]:
    config_path = _scenario_config_path(row)
    if config_path is None:
        return {}
    return _read_json(config_path)


def _input_features(row: dict[str, str]) -> dict[str, Any]:
    try:
        parsed = json.loads(row.get("features", "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _input_probability_summary(inputs: list[dict[str, str]]) -> dict[str, float | None]:
    values = [
        _as_float(row, "baseline_probability_high_risk")
        for row in inputs
        if row.get("baseline_probability_high_risk") not in {None, ""}
    ]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _load_model_artifacts() -> ModelArtifacts | None:
    settings = Settings()
    try:
        return load_artifacts(
            model_path=settings.model_path,
            columns_path=settings.columns_path,
            reference_data_path=settings.reference_data_path,
            feature_registry_path=settings.feature_registry_path,
        )
    except Exception:
        return None


def _feature_bound(
    *,
    feature_name: str,
    artifacts: ModelArtifacts,
) -> tuple[float | None, float | None]:
    feature = artifacts.feature_registry.get(feature_name)
    lower = None if feature is None else feature.global_min
    upper = None if feature is None else feature.global_max
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def _coerce_model_frame(features: dict[str, Any], artifacts: ModelArtifacts) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for column in artifacts.feature_columns:
        value = features[column]
        row[column] = artifacts.feature_registry.coerce_value(column, value)
    return pd.DataFrame([row], columns=artifacts.feature_columns)


def _probability_low_risk(features: dict[str, Any], artifacts: ModelArtifacts) -> float:
    frame = _coerce_model_frame(features, artifacts)
    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(frame)
        return float(probabilities[0, 0])
    prediction = artifacts.model.predict(frame)
    return 1.0 if int(prediction[0]) == 0 else 0.0


def _target_reached(probability_low_risk: float, config: dict[str, Any]) -> bool:
    target_class = str(config.get("target_class", "low_risk")).strip().lower()
    threshold = float(config.get("min_target_probability", 0.5))
    if target_class in {"low", "low_risk", "0"}:
        return probability_low_risk >= threshold
    if target_class in {"high", "high_risk", "1"}:
        return (1.0 - probability_low_risk) >= threshold
    return probability_low_risk >= threshold


def _directional_value(
    *,
    feature_name: str,
    baseline_value: float,
    lower: float | None,
    upper: float | None,
    artifacts: ModelArtifacts,
) -> float | None:
    feature = artifacts.feature_registry.get(feature_name)
    if lower is None or upper is None or lower > upper:
        return None
    if feature is None or feature.preferred_direction == "any":
        return None
    if feature.preferred_direction == "increase":
        if baseline_value > upper:
            return None
        return upper
    if feature.preferred_direction == "decrease":
        if baseline_value < lower:
            return None
        return lower
    return None


def _best_any_direction_value(
    *,
    feature_name: str,
    baseline_features: dict[str, Any],
    lower: float | None,
    upper: float | None,
    artifacts: ModelArtifacts,
) -> float | None:
    if lower is None or upper is None or lower > upper:
        return None
    best_value = float(baseline_features[feature_name])
    best_probability = _probability_low_risk(baseline_features, artifacts)
    for value in [lower, upper]:
        candidate = dict(baseline_features)
        candidate[feature_name] = value
        probability = _probability_low_risk(candidate, artifacts)
        if probability > best_probability:
            best_probability = probability
            best_value = value
    return best_value


def _optimistic_candidate(
    *,
    baseline_features: dict[str, Any],
    config: dict[str, Any],
    artifacts: ModelArtifacts,
) -> dict[str, Any] | None:
    if any(column not in baseline_features for column in artifacts.feature_columns):
        return None
    candidate = {column: baseline_features[column] for column in artifacts.feature_columns}
    mutable_allowed = [str(item) for item in config.get("mutable_allowed", [])]
    for feature_name in mutable_allowed:
        if feature_name not in candidate:
            continue
        try:
            baseline_value = float(candidate[feature_name])
        except (TypeError, ValueError):
            continue
        lower, upper = _feature_bound(
            feature_name=feature_name,
            artifacts=artifacts,
        )
        value = _directional_value(
            feature_name=feature_name,
            baseline_value=baseline_value,
            lower=lower,
            upper=upper,
            artifacts=artifacts,
        )
        if value is None:
            value = _best_any_direction_value(
                feature_name=feature_name,
                baseline_features=candidate,
                lower=lower,
                upper=upper,
                artifacts=artifacts,
            )
        if value is not None:
            candidate[feature_name] = value
    return candidate


def _reachability_diagnosis(
    *,
    config: dict[str, Any],
    inputs: list[dict[str, str]],
    artifacts: ModelArtifacts | None,
) -> dict[str, Any]:
    if artifacts is None:
        return {"available": False, "reason": "artifacts_unavailable"}

    rows: list[dict[str, Any]] = []
    for row in inputs:
        baseline_features = _input_features(row)
        candidate = _optimistic_candidate(
            baseline_features=baseline_features,
            config=config,
            artifacts=artifacts,
        )
        if candidate is None:
            continue
        baseline_probability = _probability_low_risk(
            {column: baseline_features[column] for column in artifacts.feature_columns},
            artifacts,
        )
        optimistic_probability = _probability_low_risk(candidate, artifacts)
        rows.append(
            {
                "request_id": row.get("request_id", ""),
                "baseline_probability_low_risk": baseline_probability,
                "optimistic_probability_low_risk": optimistic_probability,
                "improvement": optimistic_probability - baseline_probability,
                "target_reached": _target_reached(optimistic_probability, config),
            }
        )

    reached = sum(1 for row in rows if row["target_reached"])
    improvements = [float(row["improvement"]) for row in rows]
    return {
        "available": bool(rows),
        "reason": "" if rows else "no_evaluable_inputs",
        "case_count": len(rows),
        "reachable_count": reached,
        "reachable_rate": reached / len(rows) if rows else 0.0,
        "mean_probability_improvement": (
            sum(improvements) / len(improvements) if improvements else 0.0
        ),
        "cases": rows,
    }


def _feature_constraint_summary(
    *,
    mutable_allowed: list[str],
    inputs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    input_features = [_input_features(row) for row in inputs]
    summaries: list[dict[str, Any]] = []
    for feature_name in mutable_allowed:
        values = [
            float(features[feature_name])
            for features in input_features
            if feature_name in features and features[feature_name] not in {None, ""}
        ]
        summaries.append(
            {
                "feature": feature_name,
                "input_min": min(values) if values else None,
                "input_mean": sum(values) / len(values) if values else None,
                "input_max": max(values) if values else None,
                "input_count": len(values),
            }
        )
    return summaries


def _constraint_diagnosis(
    *,
    scenario: str,
    target_row: dict[str, str],
    inputs: list[dict[str, str]],
    artifacts: ModelArtifacts | None,
) -> dict[str, Any]:
    config = _effective_config(target_row)
    mutable_allowed = [str(item) for item in config.get("mutable_allowed", [])]
    return {
        "scenario": scenario,
        "input_available": bool(inputs),
        "input_count": len(inputs),
        "mutable_allowed": mutable_allowed,
        "mutable_count": len(mutable_allowed),
        "baseline_high_risk_probability": _input_probability_summary(inputs),
        "features": _feature_constraint_summary(
            mutable_allowed=mutable_allowed,
            inputs=inputs,
        ),
        "reachability": _reachability_diagnosis(
            config=config,
            inputs=inputs,
            artifacts=artifacts,
        ),
    }


def _is_likely_unreachable_stress(
    *,
    feasible_rate: float,
    expected_infeasible: bool,
    constraint_diagnosis: dict[str, Any],
) -> bool:
    if feasible_rate > 0.0 or expected_infeasible:
        return False
    reachability = constraint_diagnosis.get("reachability", {})
    return (
        bool(reachability.get("available"))
        and float(reachability.get("reachable_rate", 0.0)) <= 0.0
    )


def _diagnosis_category(
    *,
    feasible_rate: float,
    expected_infeasible: bool,
    constraint_diagnosis: dict[str, Any],
) -> str:
    if expected_infeasible:
        return "expected_infeasible_control"
    if _is_likely_unreachable_stress(
        feasible_rate=feasible_rate,
        expected_infeasible=expected_infeasible,
        constraint_diagnosis=constraint_diagnosis,
    ):
        return "likely_unreachable_stress"
    if feasible_rate <= 0.0:
        return "problematic_low_feasibility"
    return "observed_feasible"


def _cases_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("request_id", "")): row for row in rows if row.get("request_id")}


def _case_status_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "UNKNOWN")) for row in cases)
    reason_counts = Counter(str(row.get("reason_code", "UNKNOWN")) for row in cases)
    return {
        "total_cases": len(cases),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
    }


def _overlap_summary(
    *,
    target_cases: list[dict[str, Any]],
    baseline_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    target_by_request = _cases_by_request(target_cases)
    baseline_by_request = _cases_by_request(baseline_cases)
    request_ids = sorted(set(target_by_request).intersection(baseline_by_request))
    target_feasible_baseline_infeasible = 0
    baseline_feasible_target_infeasible = 0
    both_feasible = 0
    both_infeasible = 0

    for request_id in request_ids:
        target_status = str(target_by_request[request_id].get("status"))
        baseline_status = str(baseline_by_request[request_id].get("status"))
        target_feasible = target_status == "FEASIBLE"
        baseline_feasible = baseline_status == "FEASIBLE"
        if target_feasible and baseline_feasible:
            both_feasible += 1
        elif target_feasible and not baseline_feasible:
            target_feasible_baseline_infeasible += 1
        elif baseline_feasible and not target_feasible:
            baseline_feasible_target_infeasible += 1
        else:
            both_infeasible += 1

    return {
        "overlap_cases": len(request_ids),
        "both_feasible": both_feasible,
        "both_infeasible": both_infeasible,
        "target_feasible_baseline_infeasible": target_feasible_baseline_infeasible,
        "baseline_feasible_target_infeasible": baseline_feasible_target_infeasible,
    }


def _scenario_diagnosis(
    *,
    scenario: str,
    target_engine: str,
    baseline_engine: str,
    scenario_rows: dict[str, dict[str, dict[str, str]]],
    candidate_rows: dict[str, dict[str, list[dict[str, str]]]],
    top_n: int,
    artifacts: ModelArtifacts | None,
) -> dict[str, Any]:
    target_row = scenario_rows.get(target_engine, {}).get(scenario, {})
    baseline_row = scenario_rows.get(baseline_engine, {}).get(scenario, {})
    target_cases = _load_cases_for_scenario(target_row)
    baseline_cases = _load_cases_for_scenario(baseline_row)
    target_inputs = _load_inputs_for_scenario(target_row)
    target_candidates = candidate_rows.get(target_engine, {}).get(scenario, [])

    target_feasible = _as_float(target_row, "feasible_rate")
    baseline_feasible = _as_float(baseline_row, "feasible_rate")
    target_runtime = _as_float(target_row, "mean_runtime_ms")
    baseline_runtime = _as_float(baseline_row, "mean_runtime_ms")
    reason_counts = _parse_json_counter(target_row.get("reason_counts", "{}"))
    expected_outcome = _expected_outcome(target_row)
    expected_infeasible = _is_expected_infeasible(
        feasible_rate=target_feasible,
        reason_counts=reason_counts,
        expected_outcome=expected_outcome,
    )
    violation_rate = max(
        _as_float(target_row, "immutable_violation_rate"),
        _as_float(target_row, "mutable_violation_rate"),
        _as_float(target_row, "directional_violation_rate"),
    )
    constraint_diagnosis = _constraint_diagnosis(
        scenario=scenario,
        target_row=target_row,
        inputs=target_inputs,
        artifacts=artifacts,
    )

    return {
        "scenario": scenario,
        "target_engine": target_engine,
        "baseline_engine": baseline_engine,
        "target_status": target_row.get("step_status") or "missing",
        "baseline_status": baseline_row.get("step_status") or "missing",
        "target_feasible_rate": target_feasible,
        "baseline_feasible_rate": baseline_feasible,
        "feasible_gap_vs_baseline": target_feasible - baseline_feasible,
        "target_mean_runtime_ms": target_runtime,
        "baseline_mean_runtime_ms": baseline_runtime,
        "runtime_gap_ms": target_runtime - baseline_runtime,
        "target_max_violation_rate": violation_rate,
        "expected_outcome": expected_outcome,
        "expected_infeasible": expected_infeasible,
        "diagnosis_category": _diagnosis_category(
            feasible_rate=target_feasible,
            expected_infeasible=expected_infeasible,
            constraint_diagnosis=constraint_diagnosis,
        ),
        "target_case_summary": _case_status_summary(target_cases),
        "baseline_case_summary": _case_status_summary(baseline_cases),
        "overlap_summary": _overlap_summary(
            target_cases=target_cases,
            baseline_cases=baseline_cases,
        ),
        "target_reason_counts": reason_counts,
        "target_top_changed_features": _top_changed_features(target_candidates, top_n),
        "constraint_diagnosis": constraint_diagnosis,
    }


def diagnose_engine(
    comparison_root: Path,
    *,
    target_engine: str = "ocean",
    baseline_engine: str = "dice",
    top_n: int = 8,
) -> dict[str, Any]:
    collect_results(comparison_root)
    scenario_rows = _read_csv(comparison_root / "combined" / "scenario_summary.csv")
    candidates = _read_csv(comparison_root / "combined" / "candidates.csv")
    scenario_by_engine = _scenario_rows_by_engine(scenario_rows)
    candidates_by_engine_scenario = _candidate_rows_by_engine_scenario(candidates)
    artifacts = _load_model_artifacts()

    scenarios = sorted(
        set(scenario_by_engine.get(target_engine, {})).union(
            scenario_by_engine.get(baseline_engine, {})
        )
    )
    scenario_diagnoses = [
        _scenario_diagnosis(
            scenario=scenario,
            target_engine=target_engine,
            baseline_engine=baseline_engine,
            scenario_rows=scenario_by_engine,
            candidate_rows=candidates_by_engine_scenario,
            top_n=top_n,
            artifacts=artifacts,
        )
        for scenario in scenarios
    ]

    problematic_low_feasibility = [
        item["scenario"]
        for item in scenario_diagnoses
        if item["target_status"] == "completed"
        and item["diagnosis_category"] == "problematic_low_feasibility"
    ]
    likely_unreachable = [
        item["scenario"]
        for item in scenario_diagnoses
        if item["target_status"] == "completed"
        and item["diagnosis_category"] == "likely_unreachable_stress"
    ]
    expected_low_feasibility = [
        item["scenario"]
        for item in scenario_diagnoses
        if item["target_status"] == "completed"
        and item["target_feasible_rate"] <= 0.0
        and item["expected_infeasible"]
    ]
    target_worse_than_baseline = [
        item["scenario"] for item in scenario_diagnoses if item["feasible_gap_vs_baseline"] < 0.0
    ]
    violations = [
        item["scenario"] for item in scenario_diagnoses if item["target_max_violation_rate"] > 0.0
    ]

    return {
        "comparison_root": str(comparison_root),
        "target_engine": target_engine,
        "baseline_engine": baseline_engine,
        "scenario_count": len(scenario_diagnoses),
        "low_feasibility_scenarios": problematic_low_feasibility,
        "problematic_low_feasibility_scenarios": problematic_low_feasibility,
        "likely_unreachable_scenarios": likely_unreachable,
        "expected_low_feasibility_scenarios": expected_low_feasibility,
        "target_worse_than_baseline_scenarios": target_worse_than_baseline,
        "violation_scenarios": violations,
        "scenarios": scenario_diagnoses,
    }


def build_diagnostics_report(payload: dict[str, Any]) -> str:
    target_engine = payload["target_engine"]
    baseline_engine = payload["baseline_engine"]
    lines = [
        f"# {target_engine.upper()} Diagnostics",
        "",
        f"- Comparison root: `{payload['comparison_root']}`",
        f"- Baseline engine: `{baseline_engine}`",
        f"- Scenario count: {payload['scenario_count']}",
        "",
        "## Summary",
        "",
        "- Problematic low feasibility scenarios: "
        f"{', '.join(payload['problematic_low_feasibility_scenarios']) or '-'}",
        "- Likely unreachable stress scenarios: "
        f"{', '.join(payload['likely_unreachable_scenarios']) or '-'}",
        "- Expected low feasibility controls: "
        f"{', '.join(payload['expected_low_feasibility_scenarios']) or '-'}",
        "- Worse than baseline scenarios: "
        f"{', '.join(payload['target_worse_than_baseline_scenarios']) or '-'}",
        f"- Violation scenarios: {', '.join(payload['violation_scenarios']) or '-'}",
        "",
        "## Scenario Diagnostics",
        "",
        "| Scenario | Category | Status | Feasible | Baseline feasible | Gap | Runtime ms | "
        "Baseline runtime ms | Max violation | Main reasons |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in payload["scenarios"]:
        reason_counts = json.dumps(item["target_reason_counts"], ensure_ascii=True)
        lines.append(
            "| "
            + " | ".join(
                [
                    item["scenario"],
                    item["diagnosis_category"],
                    item["target_status"],
                    _percent(float(item["target_feasible_rate"])),
                    _percent(float(item["baseline_feasible_rate"])),
                    _percent(float(item["feasible_gap_vs_baseline"])),
                    _number(float(item["target_mean_runtime_ms"])),
                    _number(float(item["baseline_mean_runtime_ms"])),
                    _number(float(item["target_max_violation_rate"])),
                    reason_counts.replace("|", "\\|"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Case Overlap", ""])
    lines.append(
        "| Scenario | Overlap | Both feasible | Both infeasible | "
        f"{target_engine} only | {baseline_engine} only |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in payload["scenarios"]:
        overlap = item["overlap_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["scenario"],
                    str(overlap["overlap_cases"]),
                    str(overlap["both_feasible"]),
                    str(overlap["both_infeasible"]),
                    str(overlap["target_feasible_baseline_infeasible"]),
                    str(overlap["baseline_feasible_target_infeasible"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Target Changed Features", ""])
    for item in payload["scenarios"]:
        lines.extend([f"### {item['scenario']}", ""])
        features = item["target_top_changed_features"]
        if not features:
            lines.append("No feasible candidate deltas.")
            lines.append("")
            continue
        lines.append("| Feature | Count |")
        lines.append("| --- | ---: |")
        for feature in features:
            lines.append(f"| {feature['feature']} | {feature['count']} |")
        lines.append("")

    lines.extend(["", "## Constraint/Search-Space Diagnostics", ""])
    problem_scenarios = {
        item["scenario"]
        for item in payload["scenarios"]
        if item["diagnosis_category"]
        in {"problematic_low_feasibility", "likely_unreachable_stress"}
    }
    if not problem_scenarios:
        lines.append("No problematic low-feasibility scenarios.")
        lines.append("")
    for item in payload["scenarios"]:
        if item["scenario"] not in problem_scenarios:
            continue
        diagnosis = item["constraint_diagnosis"]
        probability = diagnosis["baseline_high_risk_probability"]
        lines.extend(
            [
                f"### {item['scenario']}",
                "",
                f"- Input rows available: {diagnosis['input_count']}",
                f"- Mutable features: {', '.join(diagnosis['mutable_allowed']) or '-'}",
                "- Baseline high-risk probability: "
                f"min={_optional_number(probability['min'])}, "
                f"mean={_optional_number(probability['mean'])}, "
                f"max={_optional_number(probability['max'])}",
                "",
            ]
        )
        reachability = diagnosis["reachability"]
        if reachability["available"]:
            lines.extend(
                [
                    "- Optimistic reachability: "
                    f"{reachability['reachable_count']}/{reachability['case_count']} "
                    f"({_percent(float(reachability['reachable_rate']))})",
                    "- Mean optimistic low-risk probability improvement: "
                    f"{_number(float(reachability['mean_probability_improvement']))}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Optimistic reachability: unavailable ({reachability['reason']})",
                    "",
                ]
            )
        features = diagnosis["features"]
        if not features:
            lines.append("No mutable feature constraints available.")
            lines.append("")
            continue
        lines.append("| Feature | Input min | Input mean | Input max | Input count |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for feature in features:
            lines.append(
                "| "
                + " | ".join(
                    [
                        feature["feature"],
                        _optional_number(feature["input_min"]),
                        _optional_number(feature["input_mean"]),
                        _optional_number(feature["input_max"]),
                        str(feature["input_count"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def write_diagnostics(comparison_root: Path, payload: dict[str, Any]) -> dict[str, Path]:
    json_path = comparison_root / f"{payload['target_engine']}_diagnostics.json"
    report_path = comparison_root / f"{payload['target_engine']}_diagnostics.md"
    _write_json(json_path, payload)
    report_path.write_text(build_diagnostics_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one engine inside a comparison run.")
    parser.add_argument(
        "comparison_root",
        nargs="?",
        type=Path,
        help="Comparison root. Defaults to experiments/results/latest/comparison.txt.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-engine", default="ocean")
    parser.add_argument("--baseline-engine", default="dice")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    args = parser.parse_args()

    comparison_root = args.comparison_root or _latest_comparison_root(args.output_root)
    payload = diagnose_engine(
        comparison_root,
        target_engine=args.target_engine,
        baseline_engine=args.baseline_engine,
        top_n=args.top_n,
    )
    outputs = write_diagnostics(comparison_root, payload)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(f"Diagnostics report written to {outputs['report']}")
        print(f"Diagnostics JSON written to {outputs['json']}")


if __name__ == "__main__":
    main()
