from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualRequest,
    CounterfactualResponse,
    PlannerInput,
    PredictionInfo,
    ValidationSummary,
)


def _valid_payload() -> dict:
    return {
        "request_id": "req-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "bmi": 31.2, "glucose": 165}},
        "constraints": {
            "immutable_features": ["age"],
            "mutable_allowed": ["bmi", "glucose"],
            "medical_rule_set_version": "med_rule_v1",
        },
        "generation": {
            "total_cfs": 3,
            "method": "nn_search",
            "random_seed": 42,
            "timeout_ms": 5000,
        },
        "preferences": {"cost_weights": {"bmi": 1.0}, "objective_weights": {"proximity": 0.3}},
    }


def test_request_schema_valid() -> None:
    payload = _valid_payload()
    parsed = CounterfactualRequest.model_validate(payload)
    assert parsed.request_id == "req-1"
    assert parsed.constraints.mutable_allowed == ["bmi", "glucose"]


def test_overlap_mutable_and_immutable_is_rejected() -> None:
    payload = _valid_payload()
    payload["constraints"]["immutable_features"] = ["age", "bmi"]
    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)


def test_response_wire_uses_class_alias_inside_planner_input() -> None:
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
                constraint_violations=0,
            ),
        ),
    )

    wire = response.to_wire()

    assert wire["input_prediction"]["class"] == "high_risk"
    assert "class_name" not in wire["input_prediction"]
    assert wire["planner_input"]["input_prediction"]["class"] == "high_risk"
    assert wire["planner_input"]["candidate_prediction"]["class"] == "low_risk"
    assert "class_name" not in wire["planner_input"]["input_prediction"]
    assert "class_name" not in wire["planner_input"]["candidate_prediction"]
