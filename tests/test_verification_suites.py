from __future__ import annotations

from pathlib import Path

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.verification.suites import (
    DEFAULT_BACKEND_SUITES,
    VerificationSuite,
    build_backend_suite_index,
    build_suite_payload,
    load_suite_scenarios,
    select_verification_suites,
)
from diabetify_cf.verification import (
    MetricSummary,
    ScenarioAggregate,
    ScenarioExpectation,
    ScenarioRunRecord,
    VerificationReport,
    VerificationScenario,
)
from diabetify_cf.schemas import CounterfactualRequest, CounterfactualResponse, ValidationSummary


def _dummy_request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-suite",
            "model_version": "xgb_v1",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {"features": {"age": 45, "BMI": 31.2, "smoking_status": 2}},
            "constraints": {
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI", "smoking_status"],
                "must_not_change": [],
            },
        }
    )


def _dummy_response() -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="req-suite",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        model_version="xgb_v1",
        cf_engine_version="nn_engine_v1",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_compliance=True,
            medical_rules_passed=True,
        ),
    )


def _dummy_aggregate() -> ScenarioAggregate:
    scenario = VerificationScenario(
        name="suite_case",
        request=_dummy_request(),
        expectation=ScenarioExpectation(expected_status=Status.FEASIBLE),
        tags=("feasible",),
    )
    run = ScenarioRunRecord(
        scenario_name="suite_case",
        iteration=1,
        response=_dummy_response(),
        verification=VerificationReport(
            request_id="req-suite",
            response_status=Status.FEASIBLE,
            candidate_results=[],
            outcome_consistent=False,
        ),
        duration_ms=10,
        expectation_matched=True,
    )
    return ScenarioAggregate(
        scenario=scenario,
        runs=[run],
        repeatability_consistent=True,
    )


def _dummy_summary() -> MetricSummary:
    return MetricSummary(
        immutable_violation_rate=None,
        mutable_violation_rate=None,
        repeatability_rate=None,
        average_latency_ms=10.0,
        p95_latency_ms=10.0,
        total_scenarios=1,
        total_runs=1,
        total_candidates=0,
    )


def test_select_verification_suites_returns_defaults_when_unspecified() -> None:
    suites = select_verification_suites()

    assert [suite.name for suite in suites] == [suite.name for suite in DEFAULT_BACKEND_SUITES]


def test_select_verification_suites_returns_requested_subset() -> None:
    suites = select_verification_suites(("repeatability_core",))

    assert [suite.name for suite in suites] == ["repeatability_core"]


def test_select_verification_suites_rejects_unknown_suite() -> None:
    try:
        select_verification_suites(("unknown_suite",))
    except ValueError as err:
        assert "unknown verification suite(s)" in str(err)
    else:
        raise AssertionError("expected unknown suite selection to raise")


def test_load_suite_scenarios_filters_using_suite_tags() -> None:
    suite = VerificationSuite(
        name="repeatability_only",
        description="repeatability filter",
        include_tags=("repeatability",),
    )

    scenarios = load_suite_scenarios(Path("configs") / "verification", suite)

    assert [scenario.name for scenario in scenarios] == ["feasible_bmi_activity_repeatability"]


def test_build_suite_payload_includes_suite_metadata() -> None:
    suite = VerificationSuite(
        name="feasible_core",
        description="feasible filter",
        include_tags=("feasible",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_aggregate()],
        summary=_dummy_summary(),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert payload["metadata"]["suite_name"] == "feasible_core"
    assert payload["metadata"]["suite_include_tags"] == ["feasible"]
    assert payload["metadata"]["runner_mode"] == "backend_suite_member"


def test_build_backend_suite_index_contains_suite_rows() -> None:
    payload = build_backend_suite_index(
        backend_base_url="http://localhost:8080",
        suite_reports=[
            {
                "suite_name": "feasible_core",
                "scenario_count": 2,
                "total_runs": 2,
                "passed": True,
                "report_path": "reports/feasible_core.json",
            }
        ],
        output_dir=Path("reports"),
        overall_summary=_dummy_summary(),
    )

    assert payload["runner_mode"] == "backend_suite"
    assert payload["backend_base_url"] == "http://localhost:8080"
    assert payload["suite_count"] == 1
    assert payload["suites"][0]["suite_name"] == "feasible_core"
    assert payload["overall_summary"]["total_scenarios"] == 1
