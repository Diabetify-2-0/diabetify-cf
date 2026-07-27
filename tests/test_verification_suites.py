from __future__ import annotations

from pathlib import Path

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    PlannerFeatureChange,
    PlannerInput,
    PredictionInfo,
    ValidationSummary,
)
from diabetify_cf.verification import (
    MetricSummary,
    ScenarioAggregate,
    ScenarioExpectation,
    ScenarioRunRecord,
    VerificationReport,
    VerificationScenario,
)
from diabetify_cf.verification.suites import (
    DEFAULT_BACKEND_SUITES,
    VerificationSuite,
    build_backend_suite_index,
    build_suite_payload,
    load_suite_scenarios,
    select_verification_suites,
)


def _dummy_request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-suite",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {"features": {"age": 45, "BMI": 31.2, "smoking_status": 2}},
            "constraints": {
                "mutable_allowed": ["BMI", "smoking_status"],
            },
        }
    )


def _dummy_response() -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="req-suite",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
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


def _dummy_candidate_response() -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="req-suite",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        candidate=CounterfactualCandidate(
            candidate_id="cand-1",
            features={
                "BMI": 23.1,
                "age": 45,
                "smoking_status": 0,
            },
            delta={
                "BMI": -8.1,
                "smoking_status": -2,
            },
            prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.77),
            metrics=CandidateMetrics(
                distance_l1=10.1,
                changed_feature_count=2,
                lof_score=1.12,
            ),
        ),
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
        planner_input=PlannerInput(
            recommended_candidate_id="cand-1",
            candidate_prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.77),
            candidate_metrics=CandidateMetrics(
                distance_l1=10.1,
                changed_feature_count=2,
                lof_score=1.12,
            ),
            changed_features=[
                PlannerFeatureChange(
                    feature_name="BMI",
                    baseline_value=31.2,
                    candidate_value=23.1,
                    delta=-8.1,
                    direction="decrease",
                ),
                PlannerFeatureChange(
                    feature_name="smoking_status",
                    baseline_value=2,
                    candidate_value=0,
                    delta=-2,
                    direction="decrease",
                ),
            ],
            mutable_allowed=["BMI", "smoking_status"],
        ),
    )


def _dummy_repeatability_aggregate() -> ScenarioAggregate:
    scenario = VerificationScenario(
        name="repeatability_case",
        request=_dummy_request(),
        expectation=ScenarioExpectation(expected_status=Status.FEASIBLE),
        repeat_count=4,
        tags=("repeatability",),
    )
    runs = [
        ScenarioRunRecord(
            scenario_name="repeatability_case",
            iteration=1,
            response=_dummy_candidate_response(),
            verification=VerificationReport(
                request_id="req-suite",
                response_status=Status.FEASIBLE,
                candidate_results=[],
                outcome_consistent=True,
            ),
            duration_ms=12,
            expectation_matched=True,
        ),
        ScenarioRunRecord(
            scenario_name="repeatability_case",
            iteration=2,
            response=_dummy_candidate_response(),
            verification=VerificationReport(
                request_id="req-suite",
                response_status=Status.FEASIBLE,
                candidate_results=[],
                outcome_consistent=True,
            ),
            duration_ms=14,
            expectation_matched=True,
        ),
        ScenarioRunRecord(
            scenario_name="repeatability_case",
            iteration=3,
            response=_dummy_candidate_response(),
            verification=VerificationReport(
                request_id="req-suite",
                response_status=Status.FEASIBLE,
                candidate_results=[],
                outcome_consistent=True,
            ),
            duration_ms=16,
            expectation_matched=True,
        ),
        ScenarioRunRecord(
            scenario_name="repeatability_case",
            iteration=4,
            response=_dummy_candidate_response(),
            verification=VerificationReport(
                request_id="req-suite",
                response_status=Status.FEASIBLE,
                candidate_results=[],
                outcome_consistent=True,
            ),
            duration_ms=18,
            expectation_matched=True,
        ),
    ]
    return ScenarioAggregate(
        scenario=scenario,
        runs=runs,
        repeatability_consistent=True,
    )


def _dummy_summary() -> MetricSummary:
    return MetricSummary(
        immutable_violation_rate=None,
        mutable_violation_rate=None,
        average_lof_score=None,
        min_lof_score=None,
        maximum_lof_score=None,
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
    suites = select_verification_suites(("plausibility_core",))

    assert [suite.name for suite in suites] == ["plausibility_core"]


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
        include_tags=("consistency_profile",),
    )

    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)

    assert len(scenarios) == 12
    assert scenarios[0].name == "consistency_profile_01"
    assert scenarios[-1].name == "consistency_profile_12"
    assert all(scenario.repeat_count == 10 for scenario in scenarios)


def test_actionability_suite_collects_all_configured_actionability_scenarios() -> None:
    suite = VerificationSuite(
        name="actionability_only",
        description="actionability filter",
        include_tags=("actionability_profile",),
    )

    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)

    assert [scenario.name for scenario in scenarios] == [
        "actionability_non_bmi_only",
        "actionability_non_bmi_hypertension",
        "actionability_non_bmi_cholesterol",
        "actionability_non_bmi_activity_hypertension",
        "actionability_non_full_subset",
        "actionability_smoker_bmi_only",
        "actionability_smoker_bmi_smoking_allowed",
        "actionability_smoker_bmi_hypertension_smoking_allowed",
        "actionability_smoker_bmi_activity_hypertension_smoking_allowed",
        "actionability_smoker_full_subset",
    ]
    assert all(scenario.repeat_count == 1 for scenario in scenarios)


def test_plausibility_suite_collects_feasible_non_repeatability_scenarios() -> None:
    suite = VerificationSuite(
        name="plausibility_only",
        description="plausibility filter",
        include_tags=("plausibility",),
    )

    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)
    names = [scenario.name for scenario in scenarios]

    assert len(names) == 25
    assert names[0] == "plausibility_full_actionable_profile_01"
    assert names[-1] == "plausibility_smoker_full_actionable_profile_25"
    assert all("plausibility" in scenario.tags for scenario in scenarios)


def test_latency_suite_collects_configured_latency_scenarios() -> None:
    suite = VerificationSuite(
        name="latency_core",
        description="latency filter",
        include_tags=("latency_profile",),
    )

    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)

    assert len(scenarios) == 25
    assert scenarios[0].name == "latency_profile_01"
    assert scenarios[-1].name == "latency_profile_25"
    assert all(scenario.repeat_count == 5 for scenario in scenarios)
    assert all("latency_profile" in scenario.tags for scenario in scenarios)


def test_build_suite_payload_includes_suite_metadata() -> None:
    suite = VerificationSuite(
        name="custom_suite",
        description="custom filter",
        include_tags=("custom",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_aggregate()],
        summary=_dummy_summary(),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert payload["metadata"]["suite_name"] == "custom_suite"
    assert payload["metadata"]["suite_include_tags"] == ["custom"]
    assert payload["metadata"]["runner_mode"] == "backend_suite_member"


def test_build_actionability_suite_payload_uses_compact_profile_shape() -> None:
    suite = VerificationSuite(
        name="actionability_core",
        description="actionability filter",
        include_tags=("actionability",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_aggregate()],
        summary=_dummy_summary(),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert "metadata" not in payload
    assert set(payload["summary"]) == {
        "immutable_violation_rate",
        "mutable_violation_rate",
        "total_scenarios",
        "total_candidates",
        "average_latency_ms",
        "p95_latency_ms",
    }
    scenario_payload = payload["scenarios"][0]
    assert scenario_payload["description"] == "Mutable: BMI, Smoking Status"
    assert scenario_payload["immutable_violation"] is False
    assert scenario_payload["mutable_violation"] is False
    assert scenario_payload["instance_profile"]["BMI"] == 31.2
    assert scenario_payload["counterfactual_candidate"] is None
    assert scenario_payload["changed_features"] is None


def test_build_plausibility_suite_payload_uses_compact_lof_shape() -> None:
    suite = VerificationSuite(
        name="plausibility_core",
        description="plausibility filter",
        include_tags=("feasible",),
        exclude_tags=("repeatability",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_aggregate()],
        summary=_dummy_summary(),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert "metadata" not in payload
    assert set(payload["summary"]) == {
        "average_lof_score",
        "min_lof_score",
        "maximum_lof_score",
        "total_scenarios",
        "total_candidates",
        "average_latency_ms",
        "p95_latency_ms",
    }
    scenario_payload = payload["scenarios"][0]
    assert scenario_payload == {
        "name": "suite_case",
        "description": "Mutable: BMI, Smoking Status",
        "lof_score": None,
    }


def test_build_repeatability_suite_payload_uses_compact_repeatability_shape() -> None:
    suite = VerificationSuite(
        name="repeatability_core",
        description="repeatability filter",
        include_tags=("repeatability",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_repeatability_aggregate()],
        summary=MetricSummary(
            immutable_violation_rate=None,
            mutable_violation_rate=None,
            average_lof_score=None,
            min_lof_score=None,
            maximum_lof_score=None,
            repeatability_rate=1.0,
            average_latency_ms=15.0,
            p95_latency_ms=17.7,
            total_scenarios=1,
            total_runs=4,
            total_candidates=0,
        ),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert "metadata" not in payload
    assert set(payload["summary"]) == {
        "repeatability_rate",
        "fully_repeatable",
        "total_scenarios",
        "total_runs",
        "average_latency_ms",
        "p95_latency_ms",
    }
    assert payload["summary"]["fully_repeatable"] is True
    scenario_payload = payload["scenarios"][0]
    assert scenario_payload["name"] == "repeatability_case"
    assert scenario_payload["description"] == "Mutable: BMI, Smoking Status"
    assert scenario_payload["repeat_count"] == 4
    assert scenario_payload["repeatable"] is True
    assert scenario_payload["all_statuses_identical"] is True
    assert scenario_payload["all_candidates_identical"] is True
    assert list(scenario_payload).index("changed_features") < list(scenario_payload).index(
        "counterfactual_profile_run_1"
    )
    assert scenario_payload["changed_features"] == {
        "BMI": {"before": 31.2, "after": 23.1},
        "smoking_status": {"before": 2, "after": 0},
    }
    assert scenario_payload["counterfactual_profile_run_1"] == {
        "age": 45,
        "BMI": 23.1,
        "smoking_status": 0,
    }
    assert scenario_payload["counterfactual_profile_run_2"] == {
        "age": 45,
        "BMI": 23.1,
        "smoking_status": 0,
    }
    assert scenario_payload["counterfactual_profile_run_3"] == {
        "age": 45,
        "BMI": 23.1,
        "smoking_status": 0,
    }
    assert "counterfactual_profile_run_4" not in scenario_payload


def test_build_latency_suite_payload_uses_latency_shape() -> None:
    suite = VerificationSuite(
        name="latency_core",
        description="latency filter",
        include_tags=("latency",),
    )

    payload = build_suite_payload(
        suite=suite,
        aggregates=[_dummy_repeatability_aggregate()],
        summary=MetricSummary(
            immutable_violation_rate=None,
            mutable_violation_rate=None,
            average_lof_score=None,
            min_lof_score=None,
            maximum_lof_score=None,
            repeatability_rate=None,
            average_latency_ms=15.0,
            p95_latency_ms=17.7,
            total_scenarios=1,
            total_runs=4,
            total_candidates=0,
        ),
        metadata={"runner_mode": "backend_suite_member"},
    )

    assert "metadata" not in payload
    assert payload["summary"] == {
        "target_average_latency_ms": 5000.0,
        "passed": True,
        "average_latency_ms": 15.0,
        "min_latency_ms": 12,
        "max_latency_ms": 18,
        "p95_latency_ms": 17.7,
        "total_scenarios": 1,
        "total_runs": 4,
        "successful_runs": 4,
        "failed_runs": 0,
    }
    scenario_payload = payload["scenarios"][0]
    assert scenario_payload["name"] == "repeatability_case"
    assert scenario_payload["repeat_count"] == 4
    assert scenario_payload["average_latency_ms"] == 15
    assert scenario_payload["min_latency_ms"] == 12
    assert scenario_payload["max_latency_ms"] == 18
    assert scenario_payload["p95_latency_ms"] == 17.7
    assert scenario_payload["terminal_statuses"] == {"FEASIBLE": 4}
    assert [run["duration_ms"] for run in scenario_payload["runs"]] == [12, 14, 16, 18]


def test_build_backend_suite_index_contains_suite_rows() -> None:
    payload = build_backend_suite_index(
        backend_base_url="http://localhost:8080",
        suite_reports=[
            {
                "suite_name": "actionability_core",
                "scenario_count": 2,
                "total_runs": 2,
                "passed": True,
                "report_path": "reports/actionability_core.json",
            }
        ],
        output_dir=Path("reports"),
        overall_summary=_dummy_summary(),
    )

    assert payload["runner_mode"] == "backend_suite"
    assert payload["backend_base_url"] == "http://localhost:8080"
    assert payload["suite_count"] == 1
    assert payload["suites"][0]["suite_name"] == "actionability_core"
    assert payload["overall_summary"]["total_scenarios"] == 1
    assert payload["overall_summary"]["average_lof_score"] is None
    assert payload["overall_summary"]["min_lof_score"] is None
    assert payload["overall_summary"]["maximum_lof_score"] is None
