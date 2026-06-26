from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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


def _valid_payload() -> dict:
    return {
        "request_id": "req-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {
            "features": {
                "age": 45,
                "BMI": 31.2,
                "is_cholesterol": 1,
            }
        },
        "constraints": {
            "mutable_allowed": ["BMI", "is_cholesterol"],
        },
        "generation": {
            "timeout_ms": 5000,
        },
    }


def test_request_schema_valid() -> None:
    payload = _valid_payload()
    parsed = CounterfactualRequest.model_validate(payload)
    assert parsed.request_id == "req-1"
    assert parsed.constraints.mutable_allowed == ["BMI", "is_cholesterol"]


def test_request_schema_rejects_legacy_immutable_features_field() -> None:
    payload = _valid_payload()
    payload["constraints"]["immutable_features"] = ["age"]
    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)


def test_request_schema_rejects_legacy_patient_id_field() -> None:
    payload = _valid_payload()
    payload["patient_id"] = "patient-123"

    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)


def test_response_wire_uses_clean_input_and_candidate_contract() -> None:
    response = CounterfactualResponse(
        request_id="req-1",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="ok",
        input_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.25),
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=True,
        ),
        candidate=CounterfactualCandidate(
            candidate_id="cand_1",
            features={"age": 45, "BMI": 27.0},
            delta={"BMI": -4.2},
            prediction=PredictionInfo(
                class_name="low_risk",
                probability_low_risk=0.72,
            ),
            metrics=CandidateMetrics(
                distance_l1=0.1,
                changed_feature_count=1,
                lof_score=1.0,
            ),
        ),
        planner_input=PlannerInput(
            input_prediction=PredictionInfo(
                class_name="high_risk",
                probability_low_risk=0.25,
            ),
            candidate_prediction=PredictionInfo(
                class_name="low_risk",
                probability_low_risk=0.72,
            ),
            candidate_metrics=CandidateMetrics(
                distance_l1=0.1,
                changed_feature_count=1,
                lof_score=1.0,
            ),
            changed_features=[
                PlannerFeatureChange(
                    feature_name="BMI",
                    baseline_value=31.2,
                    candidate_value=27.0,
                    delta=-4.2,
                    direction="decrease",
                )
            ],
            mutable_allowed=["BMI"],
        ),
    )

    wire = response.to_wire()

    assert wire["input"]["class"] == "high_risk"
    assert wire["input"]["probability_low_risk"] == 0.25
    assert wire["input"]["mutable_allowed"] == ["BMI"]
    assert wire["candidate"]["candidate_prediction"]["class"] == "low_risk"
    assert wire["candidate"]["lof_score"] == 1.0
    assert wire["candidate"]["changed_features"][0]["feature_name"] == "BMI"
    assert "planner_input" not in wire
    assert "input_prediction" not in wire
    assert "metrics" not in wire["candidate"]
    assert "delta" not in wire["candidate"]


def test_wire_response_round_trips_into_internal_model() -> None:
    wire_payload = {
        "request_id": "req-1",
        "status": "FEASIBLE",
        "reason_code": "OK",
        "message": "ok",
        "runtime_ms": 100,
        "input": {
            "class": "high_risk",
            "probability_low_risk": 0.25,
            "mutable_allowed": ["BMI"],
        },
        "candidate": {
            "candidate_id": "cand_1",
            "features": {"age": 45, "BMI": 27.0},
            "candidate_prediction": {
                "class": "low_risk",
                "probability_low_risk": 0.72,
            },
            "lof_score": 1.0,
            "changed_features": [
                {
                    "feature_name": "BMI",
                    "baseline_value": 31.2,
                    "candidate_value": 27.0,
                    "delta": -4.2,
                    "direction": "decrease",
                }
            ],
            "validation": {
                "immutable_violation": False,
                "mutable_violation": False,
                "medical_rules_passed": True,
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    parsed = CounterfactualResponse.model_validate(wire_payload)

    assert parsed.input_prediction is not None
    assert parsed.input_prediction.class_name == "high_risk"
    assert parsed.candidate is not None
    assert parsed.candidate.prediction.class_name == "low_risk"
    assert parsed.candidate.metrics.lof_score == 1.0
    assert parsed.planner_input.mutable_allowed == ["BMI"]
    assert parsed.planner_input.changed_features[0].feature_name == "BMI"
    assert parsed.validation.medical_rules_passed
