from datetime import datetime, timezone

import pandas as pd

from diabetify_cf.engine import DiceCounterfactualEngine
from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    FeatureBound,
    PlannerInput,
    PredictionInfo,
    PrescriptivePlan,
)


def _request_payload(mutable_allowed: list[str]) -> dict:
    return {
        "request_id": "req-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "xgb_v3",
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "bmi": 31.2, "glucose": 165}},
        "constraints": {
            "immutable_features": ["age"],
            "mutable_allowed": mutable_allowed,
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


def test_engine_returns_infeasible_when_no_mutable_feature() -> None:
    req = CounterfactualRequest.model_validate(_request_payload([]))
    engine = DiceCounterfactualEngine()

    result = engine.generate(req)

    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.NO_MUTABLE_FEATURE
    assert result.candidates == []


def test_engine_returns_feasible_when_input_already_satisfies_target() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = DiceCounterfactualEngine()
    engine.artifacts = object()  # type: ignore[assignment]

    def _prepared_request(request: CounterfactualRequest) -> object:
        return type(
            "Prepared",
            (),
            {
                "registry": FeatureRegistry.from_columns(["age", "bmi", "glucose"]),
                "model_columns": ["age", "bmi", "glucose"],
                "instance_features": {"age": 45, "bmi": 24.0, "glucose": 100},
                "immutable_set": {"age"},
                "mutable_allowed": ["bmi"],
                "query_df": pd.DataFrame([{"age": 45, "bmi": 24.0, "glucose": 100.0}]),
                "base_prediction": PredictionInfo(
                    class_name="low_risk",
                    probability_low_risk=0.82,
                ),
                "permitted_range": {"bmi": [20.0, 29.0]},
            },
        )()

    engine._prepare_request = _prepared_request  # type: ignore[method-assign]

    result = engine.generate(req)

    assert result.status == Status.FEASIBLE
    assert result.reason_code == ReasonCode.TARGET_ALREADY_SATISFIED
    assert result.candidates == []
    assert result.input_prediction is not None
    assert result.input_prediction.class_name == "low_risk"
    assert result.planner_input.mutable_allowed == ["bmi"]


def test_engine_returns_not_ready_when_artifacts_are_missing() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = DiceCounterfactualEngine()

    result = engine.generate(req)

    assert result.status == Status.ERROR
    assert result.reason_code == ReasonCode.ENGINE_NOT_READY


def test_as_dice_input_df_coerces_numeric_columns() -> None:
    registry = FeatureRegistry.from_columns(["feature_a", "feature_b"])
    artifacts = ModelArtifacts(
        model=object(),
        feature_columns=["feature_a", "feature_b"],
        reference_data=pd.DataFrame({"feature_a": [0.0], "feature_b": [0.0]}),
        feature_registry=registry,
        lof_model=None,
    )
    engine = DiceCounterfactualEngine()
    engine.artifacts = artifacts

    raw = pd.DataFrame({"feature_a": ["1.5"], "feature_b": ["2"]})
    typed = engine._as_dice_input_df(raw)

    assert str(typed["feature_a"].dtype) == "float64"
    assert str(typed["feature_b"].dtype) == "float64"
    assert typed["feature_a"].iloc[0] == 1.5
    assert typed["feature_b"].iloc[0] == 2.0


def test_permitted_range_includes_binary_mutable_feature() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="is_hypertension",
                feature_type="binary",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=0,
                global_max=1,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=["bmi"],
            ),
        ],
    )
    payload = _request_payload(["is_hypertension", "BMI"])
    payload["instance"]["features"] = {"is_hypertension": 1, "BMI": 31.2}
    payload["constraints"]["feature_bounds"] = {
        "is_hypertension": {"min": 0, "max": 1},
        "BMI": {"min": 18.5, "max": 35},
    }
    request = CounterfactualRequest.model_validate(payload)
    engine = DiceCounterfactualEngine()

    permitted = engine._build_permitted_range(
        model_columns=["is_hypertension", "BMI"],
        mutable_allowed=["is_hypertension", "BMI"],
        request=request,
        registry=registry,
        baseline_features={"is_hypertension": 1, "BMI": 31.2},
    )

    assert permitted["is_hypertension"] == [0.0, 1.0]
    assert permitted["BMI"] == [18.5, 31.2]


def test_permitted_range_applies_directional_constraints() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="moderate_physical_activity_frequency",
                feature_type="ordinal",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=0,
                global_max=14,
                cost_weight=1.0,
                preferred_direction="increase",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
        ],
    )
    payload = _request_payload(["moderate_physical_activity_frequency", "BMI"])
    payload["instance"]["features"] = {
        "moderate_physical_activity_frequency": 2,
        "BMI": 31.2,
    }
    request = CounterfactualRequest.model_validate(payload)
    engine = DiceCounterfactualEngine()

    permitted = engine._build_permitted_range(
        model_columns=["moderate_physical_activity_frequency", "BMI"],
        mutable_allowed=["moderate_physical_activity_frequency", "BMI"],
        request=request,
        registry=registry,
        baseline_features={"moderate_physical_activity_frequency": 2, "BMI": 31.2},
    )

    assert permitted["moderate_physical_activity_frequency"][0] == 2.0
    assert permitted["BMI"][1] == 31.2


def test_target_satisfied_enforces_min_probability() -> None:
    engine = DiceCounterfactualEngine()
    low_prediction = PredictionInfo(class_name="low_risk", probability_low_risk=0.72)
    high_prediction = PredictionInfo(class_name="high_risk", probability_low_risk=0.22)

    assert engine._target_satisfied(low_prediction, "low_risk", 0.70)
    assert not engine._target_satisfied(low_prediction, "low_risk", 0.80)
    assert engine._target_satisfied(high_prediction, "high_risk", 0.75)
    assert not engine._target_satisfied(high_prediction, "high_risk", 0.90)


def test_build_mutable_allowed_filters_immutable_non_actionable_and_duplicates() -> None:
    registry = FeatureRegistry(
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
                cost_weight=1.0,
                preferred_direction="any",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
            FeatureDefinition(
                name="smoking_status",
                feature_type="ordinal",
                immutable=False,
                actionable=False,
                default_mutable=False,
                global_min=0,
                global_max=2,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
        ],
    )
    engine = DiceCounterfactualEngine()
    mutable = engine._build_mutable_allowed(
        mutable_input=["age", "BMI", "BMI", "smoking_status", "unknown_feature"],
        model_columns=["age", "BMI", "smoking_status"],
        immutable_set={"age"},
        registry=registry,
    )

    assert mutable == ["BMI"]


def test_directional_ok_rejects_physical_activity_decrease() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="moderate_physical_activity_frequency",
                feature_type="ordinal",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=0,
                global_max=14,
                cost_weight=1.0,
                preferred_direction="increase",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
        ],
    )
    engine = DiceCounterfactualEngine()

    baseline = {"moderate_physical_activity_frequency": 2, "BMI": 31.2}
    invalid_candidate = {"moderate_physical_activity_frequency": 1, "BMI": 28.0}
    valid_candidate = {"moderate_physical_activity_frequency": 3, "BMI": 28.0}

    assert not engine._directional_ok(
        candidate=invalid_candidate,
        baseline=baseline,
        mutable_allowed={"moderate_physical_activity_frequency", "BMI"},
        registry=registry,
    )
    assert engine._directional_ok(
        candidate=valid_candidate,
        baseline=baseline,
        mutable_allowed={"moderate_physical_activity_frequency", "BMI"},
        registry=registry,
    )


def test_bounds_ok_respects_request_feature_bounds() -> None:
    payload = _request_payload(["bmi"])
    request = CounterfactualRequest.model_validate(payload)
    registry = FeatureRegistry.from_columns(["age", "bmi", "glucose"])
    engine = DiceCounterfactualEngine()

    assert engine._bounds_ok(
        {"age": 45, "bmi": 28.5, "glucose": 165},
        request,
        registry,
    )
    assert not engine._bounds_ok(
        {"age": 45, "bmi": 30.5, "glucose": 165},
        request,
        registry,
    )


class _DummyPlanner(PrescriptivePlanner):
    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate,
        planner_input: PlannerInput,
    ) -> PrescriptivePlan:
        assert planner_input.recommended_candidate_id == candidate.candidate_id
        return PrescriptivePlan(
            generation_mode="template",
            provider="dummy_test",
            clinical_scope="decision_support",
            policy_version="dummy_policy",
            summary=f"Plan for {request.request_id}",
            goals=["goal"],
            action_steps=["step"],
            safety_notes=["note"],
            monitoring_plan=["monitor"],
            missing_context=["context"],
            contraindication_flags=[],
            human_review_required=True,
        )


def test_prescriptive_plan_builder_uses_configured_planner() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = DiceCounterfactualEngine(
        planner=_DummyPlanner(),
    )
    candidate = CounterfactualCandidate(
        candidate_id="cf_1",
        features={"age": 45, "bmi": 28.0},
        delta={"bmi": -3.2},
        prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.75),
        metrics=CandidateMetrics(
            distance_l1=0.1,
            changed_feature_count=1,
            lof_score=1.0,
            constraint_violations=0,
        ),
    )

    planner_input = PlannerInput(
        recommended_candidate_id="cf_1",
        target_deltas={"bmi": -3.2},
    )
    result = engine._build_prescriptive_plan(
        request=req,
        candidate=candidate,
        planner_input=planner_input,
    )

    assert result is not None
    assert result.provider == "dummy_test"


def test_infeasible_response_preserves_request_context_in_planner_input() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = DiceCounterfactualEngine()
    prepared = type(
        "Prepared",
        (),
        {
            "registry": FeatureRegistry.from_columns(["age", "bmi", "glucose"]),
            "model_columns": ["age", "bmi", "glucose"],
            "instance_features": {"age": 45, "bmi": 31.2, "glucose": 165},
            "immutable_set": {"age"},
            "mutable_allowed": ["bmi"],
            "query_df": pd.DataFrame([{"age": 45, "bmi": 31.2, "glucose": 165.0}]),
            "base_prediction": PredictionInfo(class_name="high_risk", probability_low_risk=0.2),
            "permitted_range": {"bmi": [20.0, 29.0]},
        },
    )()

    result = engine._process_candidates(
        request=req,
        prepared=prepared,
        raw_candidates=pd.DataFrame(),
        started=0.0,
    )

    assert result.status == Status.INFEASIBLE
    assert result.input_prediction is not None
    assert result.planner_input.input_prediction is not None
    assert result.planner_input.input_prediction.class_name == "high_risk"
    assert result.planner_input.mutable_allowed == ["bmi"]
    assert result.planner_input.immutable_features == ["age"]
