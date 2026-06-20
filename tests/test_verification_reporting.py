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
    build_report_payload,
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


def _artifacts() -> ModelArtifacts:
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
        lof_model=FakeLOFModel(score=-1.1),
    )


def _request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-report",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {"features": {"age": 45, "BMI": 31.2, "smoking_status": 2}},
            "constraints": {
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI", "smoking_status"],
            },
        }
    )


def _response() -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="req-report",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
        candidate=CounterfactualCandidate(
            candidate_id="cf_1",
            features={"age": 45, "BMI": 27.0, "smoking_status": 2},
            delta={"BMI": -4.2},
            prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.8),
            metrics=CandidateMetrics(
                distance_l1=0.1,
                changed_feature_count=1,
                lof_score=1.1,
            ),
        ),
    )


@dataclass
class StubEngine(CounterfactualEngine):
    response: CounterfactualResponse

    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        return self.response


def test_build_report_payload_contains_summary_and_run_details() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    runner = ScenarioRunner(engine=StubEngine(_response()), verifier=verifier)
    scenario = VerificationScenario(
        name="report_case",
        request=_request(),
        expectation=ScenarioExpectation(expected_status=Status.FEASIBLE),
        description="report test",
        tags=("report", "verification"),
    )

    aggregates = runner.run([scenario])
    summary = runner.summarize(aggregates)
    payload = build_report_payload(
        aggregates=aggregates,
        summary=summary,
        metadata={"runner_mode": "service", "selected_tags": ["report"]},
    )

    assert payload["metadata"]["runner_mode"] == "service"
    assert payload["metadata"]["selected_tags"] == ["report"]
    assert "generated_at" in payload["metadata"]
    assert payload["scenarios"][0]["name"] == "report_case"
    assert payload["scenarios"][0]["runs"][0]["response_status"] == "FEASIBLE"
    assert payload["scenarios"][0]["runs"][0]["verification"]["candidates"][0]["candidate_id"] == "cf_1"
