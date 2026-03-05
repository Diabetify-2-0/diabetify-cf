from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from diabetify_cf.schemas import CounterfactualRequest


def _valid_payload() -> dict:
    return {
        "request_id": "req-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "xgb_v3",
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "bmi": 31.2, "glucose": 165}},
        "constraints": {
            "immutable_features": ["age"],
            "mutable_allowed": ["bmi", "glucose"],
            "feature_bounds": {"bmi": {"min": 20.0, "max": 29.0}},
            "must_not_change": [],
            "medical_rule_set_version": "med_rule_v1",
        },
        "generation": {
            "total_cfs": 3,
            "method": "dice_genetic",
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


def test_feature_bound_invalid_when_min_greater_than_max() -> None:
    payload = _valid_payload()
    payload["constraints"]["feature_bounds"]["bmi"] = {"min": 30.0, "max": 20.0}
    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)


def test_overlap_mutable_and_immutable_is_rejected() -> None:
    payload = _valid_payload()
    payload["constraints"]["immutable_features"] = ["age", "bmi"]
    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)
