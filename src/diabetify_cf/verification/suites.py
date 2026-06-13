from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diabetify_cf.verification.fixtures import load_verification_scenarios
from diabetify_cf.verification.reporting import build_report_payload
from diabetify_cf.verification.runner import MetricSummary, ScenarioAggregate


@dataclass(frozen=True)
class VerificationSuite:
    name: str
    description: str
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()


DEFAULT_BACKEND_SUITES: tuple[VerificationSuite, ...] = (
    VerificationSuite(
        name="feasible_core",
        description="Feasible production scenarios, including target-already-satisfied cases.",
        include_tags=("feasible",),
    ),
    VerificationSuite(
        name="infeasible_core",
        description="Infeasible production scenarios covering no-mutable, target-unreachable, and medical-only outcomes.",
        include_tags=("infeasible",),
    ),
    VerificationSuite(
        name="repeatability_core",
        description="Repeated production scenarios used to compute repeatability stability.",
        include_tags=("repeatability",),
    ),
)


def select_verification_suites(
    suite_names: tuple[str, ...] = (),
) -> list[VerificationSuite]:
    available = {suite.name: suite for suite in DEFAULT_BACKEND_SUITES}
    if not suite_names:
        return list(DEFAULT_BACKEND_SUITES)

    missing = [name for name in suite_names if name not in available]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"unknown verification suite(s): {joined}")

    return [available[name] for name in suite_names]


def load_suite_scenarios(
    scenarios_path: str | Path,
    suite: VerificationSuite,
):
    return load_verification_scenarios(
        scenarios_path,
        include_tags=suite.include_tags,
        exclude_tags=suite.exclude_tags,
    )


def build_suite_payload(
    *,
    suite: VerificationSuite,
    aggregates: list[ScenarioAggregate],
    summary: MetricSummary,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_report_payload(
        aggregates=aggregates,
        summary=summary,
        metadata={
            "suite_name": suite.name,
            "suite_description": suite.description,
            "suite_include_tags": list(suite.include_tags),
            "suite_exclude_tags": list(suite.exclude_tags),
            **(metadata or {}),
        },
    )


def build_backend_suite_index(
    *,
    backend_base_url: str,
    suite_reports: list[dict[str, Any]],
    output_dir: str | Path,
    overall_summary: MetricSummary | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    payload = {
        "runner_mode": "backend_suite",
        "backend_base_url": backend_base_url,
        "output_dir": str(root),
        "suite_count": len(suite_reports),
        "suites": suite_reports,
    }
    if overall_summary is not None:
        payload["overall_summary"] = {
            "immutable_violation_rate": overall_summary.immutable_violation_rate,
            "mutable_violation_rate": overall_summary.mutable_violation_rate,
            "externally_verified_target_satisfaction_rate": overall_summary.externally_verified_target_satisfaction_rate,
            "infeasible_handling_accuracy": overall_summary.infeasible_handling_accuracy,
            "end_to_end_scenario_pass_rate": overall_summary.end_to_end_scenario_pass_rate,
            "external_lof_verification_accuracy": overall_summary.external_lof_verification_accuracy,
            "repeatability_rate": overall_summary.repeatability_rate,
            "average_latency_ms": overall_summary.average_latency_ms,
            "p95_latency_ms": overall_summary.p95_latency_ms,
            "constraint_gate_compliance_rate": overall_summary.constraint_gate_compliance_rate,
            "total_scenarios": overall_summary.total_scenarios,
            "total_runs": overall_summary.total_runs,
            "total_candidates": overall_summary.total_candidates,
        }
    return payload
