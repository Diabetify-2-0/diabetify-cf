from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
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
from diabetify_cf.verification import ExternalCounterfactualVerifier


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


def _request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-verify",
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
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI", "smoking_status"],
            },
        }
    )


def _candidate(
    *,
    features: dict[str, float | int],
    low_risk_probability: float = 0.80,
) -> CounterfactualCandidate:
    return CounterfactualCandidate(
        candidate_id="cf_1",
        features=features,
        delta={"BMI": round(float(features["BMI"]) - 31.2, 6)},
        prediction=PredictionInfo(
            class_name="low_risk" if low_risk_probability >= 0.5 else "high_risk",
            probability_low_risk=low_risk_probability,
        ),
        metrics=CandidateMetrics(
            distance_l1=0.1,
            changed_feature_count=1,
            lof_score=1.1,
        ),
    )


def _response(candidate: CounterfactualCandidate) -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="req-verify",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
        candidate=candidate,
    )


def test_external_verifier_confirms_valid_returned_candidate() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)

    report = verifier.verify_response(
        request=_request(),
        response=_response(
            _candidate(features={"age": 45, "BMI": 27.0, "smoking_status": 2})
        ),
    )

    assert report.outcome_consistent
    assert report.immutable_violation_rate == 0.0
    assert report.mutable_violation_rate == 0.0


def test_external_verifier_detects_target_failure_independently() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)

    report = verifier.verify_response(
        request=_request(),
        response=_response(
            _candidate(
                features={"age": 45, "BMI": 35.0, "smoking_status": 2},
                low_risk_probability=0.80,
            )
        ),
    )

    assert not report.outcome_consistent
    assert report.candidate_results[0].prediction_matches_returned is False


def test_external_verifier_detects_immutable_violation_independently() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)

    report = verifier.verify_response(
        request=_request(),
        response=_response(
            _candidate(features={"age": 46, "BMI": 27.0, "smoking_status": 2})
        ),
    )

    assert not report.outcome_consistent
    assert report.immutable_violation_rate == 1.0
    assert report.candidate_results[0].immutable_ok is False


def test_external_verifier_detects_mutable_and_transition_violations() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)

    report = verifier.verify_response(
        request=_request(),
        response=_response(
            _candidate(features={"age": 45, "BMI": 27.0, "smoking_status": 0})
        ),
    )

    assert not report.outcome_consistent
    assert report.mutable_violation_rate == 0.0
    assert report.candidate_results[0].transition_ok is False


def test_external_verifier_detects_feature_change_outside_user_selected_mutable_set() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    request = CounterfactualRequest.model_validate(
        {
            "request_id": "req-verify-bmi-only",
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
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI"],
            },
        }
    )

    report = verifier.verify_response(
        request=request,
        response=_response(
            _candidate(features={"age": 45, "BMI": 27.0, "smoking_status": 1})
        ),
    )

    assert not report.outcome_consistent
    assert report.mutable_violation_rate == 1.0
    assert report.candidate_results[0].mutable_ok is False


def test_external_verifier_rejects_candidate_outside_plausibility_threshold() -> None:
    verifier = ExternalCounterfactualVerifier(
        artifacts=_artifacts(lof_score=3.4),
        max_lof_score=2.5,
    )

    report = verifier.verify_response(
        request=_request(),
        response=_response(
            _candidate(features={"age": 45, "BMI": 27.0, "smoking_status": 2})
        ),
    )

    assert not report.outcome_consistent
    assert report.candidate_results[0].external_lof_within_threshold is False


def test_external_verifier_accepts_infeasible_without_candidates() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    response = CounterfactualResponse(
        request_id="req-verify",
        status=Status.INFEASIBLE,
        reason_code=ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,
        message="no solution",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=False,
        ),
    )

    report = verifier.verify_response(request=_request(), response=response)

    assert report.outcome_consistent
    assert report.candidate_count == 0


def test_external_verifier_accepts_target_already_satisfied_without_candidates() -> None:
    verifier = ExternalCounterfactualVerifier(artifacts=_artifacts(), max_lof_score=2.5)
    healthy_request = CounterfactualRequest.model_validate(
        {
            "request_id": "req-target-already-satisfied",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {
                "features": {
                    "age": 45,
                    "BMI": 20.0,
                    "smoking_status": 0,
                }
            },
            "constraints": {
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI", "smoking_status"],
            },
        }
    )
    response = CounterfactualResponse(
        request_id="req-target-already-satisfied",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.TARGET_ALREADY_SATISFIED,
        message="Input already satisfies target class/probability; no changes are required.",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
    )

    report = verifier.verify_response(request=healthy_request, response=response)

    assert report.outcome_consistent
    assert report.candidate_count == 0
