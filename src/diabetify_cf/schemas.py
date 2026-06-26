from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diabetify_cf.reason_codes import ReasonCode, Status

JSONNumber = int | float
JSONFeatureValue = int | float | bool | str


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_class: str = "low_risk"
    min_target_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mutable_allowed: list[str] = Field(default_factory=list)


class GenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_ms: int = Field(default=5000, ge=100, le=60000)


class InstanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, JSONFeatureValue]


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target: TargetSpec = Field(default_factory=TargetSpec)
    instance: InstanceSpec
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)


class PredictionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_name: str = Field(alias="class")
    probability_low_risk: float = Field(ge=0.0, le=1.0)

    def to_wire(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "probability_low_risk": self.probability_low_risk,
        }


class CandidateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_l1: float
    changed_feature_count: int
    lof_score: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "changed_feature_count": self.changed_feature_count,
            "lof_score": self.lof_score,
        }


class CounterfactualCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    features: dict[str, JSONFeatureValue]
    delta: dict[str, JSONNumber]
    prediction: PredictionInfo
    metrics: CandidateMetrics

    @model_validator(mode="before")
    @classmethod
    def normalize_wire_candidate(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "prediction" in value or "metrics" in value or "delta" in value:
            return value

        normalized = dict(value)
        changed_features = normalized.pop("changed_features", [])
        candidate_prediction = normalized.pop("candidate_prediction", None)
        lof_score = normalized.pop("lof_score", None)
        normalized.pop("validation", None)

        if candidate_prediction is not None:
            normalized["prediction"] = candidate_prediction
        if changed_features:
            delta: dict[str, JSONNumber] = {}
            for item in changed_features:
                if isinstance(item, dict) and "feature_name" in item and "delta" in item:
                    feature_name = item.get("feature_name")
                    feature_delta = item.get("delta")
                    if isinstance(feature_name, str) and isinstance(feature_delta, (int, float)):
                        delta[feature_name] = feature_delta
            normalized["delta"] = delta
        else:
            normalized["delta"] = {}
        normalized["metrics"] = {
            "distance_l1": 0.0,
            "changed_feature_count": len(changed_features) if isinstance(changed_features, list) else 0,
            "lof_score": float(lof_score) if isinstance(lof_score, (int, float)) else 1.0,
        }
        return normalized

    def to_wire(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "features": self.features,
            "candidate_prediction": self.prediction.to_wire(),
            "lof_score": self.metrics.lof_score,
        }


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    immutable_violation: bool
    mutable_violation: bool
    medical_rules_passed: bool


class PlannerFeatureChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_name: str
    baseline_value: JSONFeatureValue
    candidate_value: JSONFeatureValue
    delta: JSONNumber
    direction: str


class PlannerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommended_candidate_id: str | None = None
    target_deltas: dict[str, JSONNumber] = Field(default_factory=dict)
    input_prediction: PredictionInfo | None = None
    candidate_prediction: PredictionInfo | None = None
    candidate_metrics: CandidateMetrics | None = None
    changed_features: list[PlannerFeatureChange] = Field(default_factory=list)
    mutable_allowed: list[str] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        if self.input_prediction is not None:
            payload["input_prediction"] = self.input_prediction.to_wire()
        if self.candidate_prediction is not None:
            payload["candidate_prediction"] = self.candidate_prediction.to_wire()
        if self.candidate_metrics is not None:
            payload["candidate_metrics"] = self.candidate_metrics.to_wire()
        return payload


class CounterfactualResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Status
    reason_code: ReasonCode
    message: str
    runtime_ms: int = 0
    input_prediction: PredictionInfo | None = None
    candidate: CounterfactualCandidate | None = None
    validation: ValidationSummary
    planner_input: PlannerInput = Field(default_factory=PlannerInput)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def normalize_wire_response(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "input" not in value and "planner_input" in value:
            return value

        normalized = dict(value)
        input_payload = normalized.pop("input", None)
        candidate_payload = normalized.get("candidate")

        if isinstance(candidate_payload, dict) and "validation" not in normalized:
            candidate_validation = candidate_payload.get("validation")
            if isinstance(candidate_validation, dict):
                normalized["validation"] = candidate_validation

        if isinstance(input_payload, dict):
            input_class = input_payload.get("class")
            probability = input_payload.get("probability_low_risk")
            if isinstance(input_class, str) and isinstance(probability, (int, float)):
                normalized["input_prediction"] = {
                    "class": input_class,
                    "probability_low_risk": probability,
                }

            planner_input: dict[str, Any] = {
                "mutable_allowed": input_payload.get("mutable_allowed", []),
            }
            if "input_prediction" in normalized:
                planner_input["input_prediction"] = normalized["input_prediction"]

            if isinstance(candidate_payload, dict):
                changed_features = candidate_payload.get("changed_features", [])
                planner_input["recommended_candidate_id"] = candidate_payload.get("candidate_id")
                planner_input["candidate_prediction"] = candidate_payload.get("candidate_prediction")
                planner_input["candidate_metrics"] = {
                    "distance_l1": 0.0,
                    "changed_feature_count": len(changed_features)
                    if isinstance(changed_features, list)
                    else 0,
                    "lof_score": candidate_payload.get("lof_score", 1.0),
                }
                planner_input["changed_features"] = changed_features
                planner_input["target_deltas"] = {
                    item["feature_name"]: item["delta"]
                    for item in changed_features
                    if isinstance(item, dict)
                    and isinstance(item.get("feature_name"), str)
                    and isinstance(item.get("delta"), (int, float))
                }

            normalized["planner_input"] = planner_input

        return normalized

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "message": self.message,
            "runtime_ms": self.runtime_ms,
            "candidate": None,
            "timestamp": self.timestamp.isoformat(),
        }

        input_payload: dict[str, Any] = {}
        if self.input_prediction is not None:
            input_payload.update(self.input_prediction.to_wire())
        if self.planner_input.mutable_allowed:
            input_payload["mutable_allowed"] = list(self.planner_input.mutable_allowed)
        if input_payload:
            payload["input"] = input_payload

        if self.candidate is not None:
            candidate_payload = self.candidate.to_wire()
            candidate_payload["changed_features"] = [
                item.model_dump() for item in self.planner_input.changed_features
            ]
            candidate_payload["validation"] = self.validation.model_dump()
            payload["candidate"] = candidate_payload
        else:
            payload["validation"] = self.validation.model_dump()
        return payload
