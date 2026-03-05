from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diabetify_cf.reason_codes import ReasonCode, Status

JSONNumber = Union[int, float]
JSONFeatureValue = Union[int, float, bool, str]


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_class: str = "low_risk"
    min_target_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class FeatureBound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "FeatureBound":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    immutable_features: List[str] = Field(default_factory=list)
    mutable_allowed: List[str] = Field(default_factory=list)
    feature_bounds: Dict[str, FeatureBound] = Field(default_factory=dict)
    must_not_change: List[str] = Field(default_factory=list)
    medical_rule_set_version: str = "med_rule_v1"

    @model_validator(mode="after")
    def validate_no_overlap(self) -> "ConstraintSpec":
        immutable_set = set(self.immutable_features)
        mutable_set = set(self.mutable_allowed)
        overlap = immutable_set.intersection(mutable_set)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"immutable and mutable overlap: {joined}")
        return self


class GenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cfs: int = Field(default=3, ge=1, le=20)
    method: str = "dice_genetic"
    random_seed: int = 42
    timeout_ms: int = Field(default=5000, ge=100, le=60000)


class PreferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_weights: Dict[str, float] = Field(default_factory=dict)
    objective_weights: Dict[str, float] = Field(default_factory=dict)


class InstanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, JSONFeatureValue]


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    patient_id: Optional[str] = None
    model_version: str = "xgb_v1"
    target: TargetSpec = Field(default_factory=TargetSpec)
    instance: InstanceSpec
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    preferences: PreferenceSpec = Field(default_factory=PreferenceSpec)


class PredictionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_name: str = Field(alias="class")
    probability_low_risk: float = Field(ge=0.0, le=1.0)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "probability_low_risk": self.probability_low_risk,
        }


class CandidateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_l1: float
    changed_feature_count: int
    lof_score: float
    constraint_violations: int


class CounterfactualCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    features: Dict[str, JSONFeatureValue]
    delta: Dict[str, JSONNumber]
    prediction: PredictionInfo
    metrics: CandidateMetrics

    def to_wire(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload["prediction"] = self.prediction.to_wire()
        return payload


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    immutable_violation: bool
    mutable_compliance: bool
    medical_rules_passed: bool


class PlannerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommended_candidate_id: Optional[str] = None
    target_deltas: Dict[str, JSONNumber] = Field(default_factory=dict)


class PrescriptivePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_mode: str
    provider: str
    summary: str
    goals: List[str] = Field(default_factory=list)
    action_steps: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    monitoring_plan: List[str] = Field(default_factory=list)
    disclaimer: str = "Panduan ini bersifat edukatif dan tidak menggantikan konsultasi dokter."


class CounterfactualResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Status
    reason_code: ReasonCode
    message: str
    model_version: str
    cf_engine_version: str
    constraint_version: str = "cf_spec_1.0.0"
    runtime_ms: int = 0
    input_prediction: Optional[PredictionInfo] = None
    candidates: List[CounterfactualCandidate] = Field(default_factory=list)
    validation: ValidationSummary
    planner_input: PlannerInput = Field(default_factory=PlannerInput)
    prescriptive_plan: Optional[PrescriptivePlan] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_wire(self) -> Dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        payload["status"] = self.status.value
        payload["reason_code"] = self.reason_code.value
        if self.input_prediction is not None:
            payload["input_prediction"] = self.input_prediction.to_wire()
        payload["candidates"] = [candidate.to_wire() for candidate in self.candidates]
        payload["timestamp"] = self.timestamp.isoformat()
        return payload
