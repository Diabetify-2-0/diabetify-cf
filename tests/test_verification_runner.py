from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    PredictionInfo,
    ValidationSummary,
)
from diabetify_cf.verification import (
    ExternalCounterfactualVerifier,
    ScenarioExpectation,
    ScenarioRunner,
    VerificationScenario,
)


def _request(request_id: str = "req-runner") -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": request_id,
            "target": {
                "target_class": "low_risk",
                "min_target_probability": 0.5,
            },
            "instance": {
                "features": {
                    "age": 45,
                    "BMI": 31.2,
                    "smoking_status": 2,
                }
            },
            "constraints": {
                "mutable_allowed": ["BMI", "smoking_status"],
            },
        }
    )


@dataclass
class FakeModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        bmi = float(frame.iloc[0]["BMI"])
        if bmi <= 28.0:
            return np.array([[0.80, 0.20]])
        return np.array([[0.20, 0.80]])


@dataclass
class FakeLOFModel:
    score: float = -1.1

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.array([self.score], dtype=float)


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="age",
                feature_type="continuous",
                immutable=True,
                actionable=False,
                default_mutable=False,
                global_min=18,
                global_max=100,
                cost_weight=10.0,
                preferred_direction="any",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10.0,
                global_max=60.0,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=["bmi"],
            ),
            FeatureDefinition(
                name="smoking_status",
                feature_type="ordinal",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=0.0,
                global_max=2.0,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
                allowed_transitions={0: (0,), 1: (1,), 2: (1, 2)},
            ),
        ],
    )


def _artifacts(lof_score: float = 1.1) -> ModelArtifacts:
    return ModelArtifacts(
        model=FakeModel(),
        feature_columns=["age", "BMI", "smoking_status"],
        reference_data=pd.DataFrame(
            [
                {"age": 45, "BMI": 31.2, "smoking_status": 2},
                {"age": 45, "BMI": 27.0, "smoking_status": 1},
            ]
        ),
        feature_registry=_registry(),
        lof_model=FakeLOFModel(score=-lof_score),
    )


def _candidate(candidate_id: str = "cf_1") -> CounterfactualCandidate:
    return CounterfactualCandidate(
        candidate_id=candidate_id,
        features={"age": 45, "BMI": 27.0, "smoking_status": 2},
        delta={"BMI": -4.2},
        prediction=PredictionInfo(
            class_name="low_risk",
            probability_low_risk=0.8,
        ),
        metrics=CandidateMetrics(
            distance_l1=0.1,
            changed_feature_count=1,
            lof_score=1.1,
        ),
    )


def _feasible_response(request_id: str = "req-runner") -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id=request_id,
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
        candidate=_candidate(),
    )


def _infeasible_response(request_id: str = "req-runner") -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id=request_id,
        status=Status.INFEASIBLE,
        reason_code=ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,
        message="no solution",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=False,
        ),
    )


@dataclass
class StubEngine(CounterfactualEngine):
    responses: list[CounterfactualResponse]

    def __post_init__(self) -> None:
        self._index = 0

    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        response = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return response


def test_scenario_runner_summarizes_feasible_and_infeasible_metrics() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    runner = ScenarioRunner(
        engine=StubEngine(
            responses=[
                _feasible_response("req-feasible"),
                _infeasible_response("req-infeasible"),
            ]
        ),
        verifier=verifier,
    )
    scenarios = [
        VerificationScenario(
            name="feasible_case",
            request=_request("req-feasible"),
            expectation=ScenarioExpectation(expected_status=Status.FEASIBLE),
        ),
        VerificationScenario(
            name="infeasible_case",
            request=_request("req-infeasible"),
            expectation=ScenarioExpectation(
                expected_status=Status.INFEASIBLE,
                expected_reason_codes=(ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,),
                no_solution_expected=True,
            ),
        ),
    ]

    aggregates = runner.run(scenarios)
    summary = runner.summarize(aggregates)

    assert len(aggregates) == 2
    assert aggregates[0].passed
    assert aggregates[1].passed
    assert summary.total_scenarios == 2
    assert summary.total_runs == 2
    assert summary.total_candidates == 1
    assert summary.immutable_violation_rate == 0.0
    assert summary.mutable_violation_rate == 0.0
    assert summary.lof_violation_rate == 0.0
    assert summary.average_lof_score == 1.1
    assert summary.min_lof_score == 1.1
    assert summary.max_lof_score == 1.1


def test_scenario_runner_measures_repeatability_from_repeated_runs() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    runner = ScenarioRunner(
        engine=StubEngine(
            responses=[
                _feasible_response("req-repeat"),
                _feasible_response("req-repeat"),
            ]
        ),
        verifier=verifier,
    )
    scenario = VerificationScenario(
        name="repeatable_case",
        request=_request("req-repeat"),
        expectation=ScenarioExpectation(expected_status=Status.FEASIBLE),
        repeat_count=2,
    )

    aggregates = runner.run([scenario])
    summary = runner.summarize(aggregates)

    assert aggregates[0].repeatability_consistent
    assert summary.lof_violation_rate == 0.0
    assert summary.average_lof_score == 1.1
    assert summary.min_lof_score == 1.1
    assert summary.max_lof_score == 1.1
    assert summary.repeatability_rate == 1.0
