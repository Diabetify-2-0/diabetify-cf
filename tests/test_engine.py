from datetime import datetime, timezone
from math import sqrt
from time import perf_counter, sleep

import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.engine.shared import ArtifactBackedCounterfactualEngine
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    PredictionInfo,
)


class TestCounterfactualEngine(ArtifactBackedCounterfactualEngine):
    __test__ = False
    engine_version = "test_engine_v1"

    def _generate_raw_candidates(self, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    def _as_test_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._as_model_input_df(frame)


def _request_payload(mutable_allowed: list[str]) -> dict:
    return {
        "request_id": "req-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "bmi": 31.2, "glucose": 165}},
        "constraints": {
            "mutable_allowed": mutable_allowed,
        },
        "generation": {
            "timeout_ms": 5000,
        },
    }


def test_engine_returns_infeasible_when_no_mutable_feature() -> None:
    req = CounterfactualRequest.model_validate(_request_payload([]))
    engine = TestCounterfactualEngine()

    result = engine.generate(req)

    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.NO_MUTABLE_FEATURE
    assert result.candidate is None


def test_engine_returns_feasible_when_input_already_satisfies_target() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = TestCounterfactualEngine()
    engine.artifacts = object()  

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
            },
        )()

    engine._prepare_request = _prepared_request  

    result = engine.generate(req)

    assert result.status == Status.FEASIBLE
    assert result.reason_code == ReasonCode.TARGET_ALREADY_SATISFIED
    assert result.candidate is None
    assert result.input_prediction is not None
    assert result.input_prediction.class_name == "low_risk"
    assert result.planner_input.mutable_allowed == ["bmi"]


def test_engine_returns_feasible_when_low_risk_probability_already_meets_requested_threshold() -> (
    None
):
    payload = _request_payload(["bmi"])
    payload["target"]["min_target_probability"] = 0.30
    req = CounterfactualRequest.model_validate(payload)
    engine = TestCounterfactualEngine()
    engine.artifacts = object()  

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
                    probability_low_risk=0.35,
                ),
            },
        )()

    engine._prepare_request = _prepared_request  

    result = engine.generate(req)

    assert result.status == Status.FEASIBLE
    assert result.reason_code == ReasonCode.TARGET_ALREADY_SATISFIED
    assert result.candidate is None
    assert result.input_prediction is not None
    assert result.input_prediction.class_name == "low_risk"


def test_engine_returns_not_ready_when_artifacts_are_missing() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = TestCounterfactualEngine()

    result = engine.generate(req)

    assert result.status == Status.ERROR
    assert result.reason_code == ReasonCode.ENGINE_NOT_READY


def test_engine_timeout_returns_terminal_response_without_waiting_for_generator() -> None:
    payload = _request_payload(["bmi"])
    payload["generation"]["timeout_ms"] = 100
    req = CounterfactualRequest.model_validate(payload)
    engine = TestCounterfactualEngine()
    engine.artifacts = object()  

    def _prepared_request(request: CounterfactualRequest) -> object:
        return type(
            "Prepared",
            (),
            {
                "registry": FeatureRegistry.from_columns(["age", "bmi", "glucose"]),
                "model_columns": ["age", "bmi", "glucose"],
                "instance_features": {"age": 45, "bmi": 31.2, "glucose": 165},
                "immutable_set": {"age"},
                "mutable_allowed": ["bmi"],
                "query_df": pd.DataFrame([{"age": 45, "bmi": 31.2, "glucose": 165.0}]),
                "base_prediction": PredictionInfo(
                    class_name="high_risk",
                    probability_low_risk=0.2,
                ),
            },
        )()

    def _slow_generate_candidates(**_kwargs: object) -> pd.DataFrame:
        sleep(0.3)
        return pd.DataFrame([{"age": 45, "bmi": 28.0, "glucose": 165.0}])

    engine._prepare_request = _prepared_request  
    engine._generate_raw_candidates = _slow_generate_candidates  
    started = perf_counter()

    result = engine.generate(req)

    assert (perf_counter() - started) < 0.25
    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION
    assert result.input_prediction is not None
    assert result.planner_input.input_prediction is not None


def test_as_model_input_df_coerces_numeric_columns() -> None:
    registry = FeatureRegistry.from_columns(["feature_a", "feature_b"])
    artifacts = ModelArtifacts(
        model=object(),
        feature_columns=["feature_a", "feature_b"],
        reference_data=pd.DataFrame({"feature_a": [0.0], "feature_b": [0.0]}),
        feature_registry=registry,
        lof_model=None,
    )
    engine = TestCounterfactualEngine()
    engine.artifacts = artifacts

    raw = pd.DataFrame({"feature_a": ["1.5"], "feature_b": ["2"]})
    typed = engine._as_test_input_df(raw)

    assert str(typed["feature_a"].dtype) == "float64"
    assert str(typed["feature_b"].dtype) == "float64"
    assert typed["feature_a"].iloc[0] == 1.5
    assert typed["feature_b"].iloc[0] == 2.0


def _bmi_registry() -> FeatureRegistry:
    return FeatureRegistry(
        version="test_v1",
        features=[
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
            )
        ],
    )


def _engine_with_bmi_artifacts() -> TestCounterfactualEngine:
    registry = _bmi_registry()
    engine = TestCounterfactualEngine()
    engine.artifacts = ModelArtifacts(
        model=object(),
        feature_columns=["BMI"],
        reference_data=pd.DataFrame({"BMI": [31.2]}),
        feature_registry=registry,
        lof_model=None,
    )
    return engine


def test_prepare_request_rejects_duplicate_alias_features() -> None:
    payload = _request_payload(["BMI"])
    payload["instance"]["features"] = {"BMI": 31.2, "bmi": 30.0}
    request = CounterfactualRequest.model_validate(payload)
    engine = _engine_with_bmi_artifacts()

    result = engine.generate(request)

    assert result.status == Status.ERROR
    assert result.reason_code == ReasonCode.INVALID_INPUT_SCHEMA
    assert result.message == "Invalid counterfactual request payload."


def test_prepare_request_rejects_instance_feature_outside_registry_range() -> None:
    payload = _request_payload(["BMI"])
    payload["instance"]["features"] = {"BMI": 70.0}
    request = CounterfactualRequest.model_validate(payload)
    engine = _engine_with_bmi_artifacts()

    result = engine.generate(request)

    assert result.status == Status.ERROR
    assert result.reason_code == ReasonCode.INVALID_INPUT_SCHEMA
    assert result.message == "Invalid counterfactual request payload."


def test_target_satisfied_enforces_min_probability() -> None:
    engine = TestCounterfactualEngine()
    low_prediction = PredictionInfo(class_name="low_risk", probability_low_risk=0.72)
    moderate_low_prediction = PredictionInfo(class_name="low_risk", probability_low_risk=0.35)

    assert engine._target_satisfied(low_prediction, "low_risk", 0.70)
    assert not engine._target_satisfied(low_prediction, "low_risk", 0.80)
    assert engine._target_satisfied(moderate_low_prediction, "low_risk", 0.30)
    assert not engine._target_satisfied(moderate_low_prediction, "low_risk", 0.40)


def test_objective_score_uses_raw_lof_as_plausibility_penalty() -> None:
    registry = FeatureRegistry.from_columns(["BMI"])
    prediction = PredictionInfo(class_name="low_risk", probability_low_risk=0.8)
    lower_lof_candidate = CounterfactualCandidate(
        candidate_id="lower-lof",
        features={"BMI": 29.0},
        delta={"BMI": -1.0},
        prediction=prediction,
        metrics=CandidateMetrics(
            distance_l1=0.0,
            changed_feature_count=1,
            lof_score=0.8,
        ),
    )
    higher_lof_candidate = CounterfactualCandidate(
        candidate_id="higher-lof",
        features={"BMI": 29.0},
        delta={"BMI": -1.0},
        prediction=prediction,
        metrics=CandidateMetrics(
            distance_l1=0.0,
            changed_feature_count=1,
            lof_score=1.2,
        ),
    )
    engine = TestCounterfactualEngine()
    engine._reference_lof_scores = lambda: pd.Series([0.9, 1.1, 1.3]).to_numpy()
    weights = {
        "proximity": 0.0,
        "plausibility": 1.0,
    }

    lower_score = engine._objective_score(
        candidate=lower_lof_candidate,
        preferences=weights,
        mutable_allowed=["BMI"],
        registry=registry,
    )
    higher_score = engine._objective_score(
        candidate=higher_lof_candidate,
        preferences=weights,
        mutable_allowed=["BMI"],
        registry=registry,
    )

    assert lower_score == 0.0
    assert higher_score == 2 / 3
    assert lower_score < higher_score


def test_objective_score_defaults_use_proximity_plausibility_weights() -> None:
    registry = FeatureRegistry.from_columns(["BMI"])
    candidate = CounterfactualCandidate(
        candidate_id="weighted",
        features={"BMI": 29.0},
        delta={"BMI": -1.0},
        prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.8),
        metrics=CandidateMetrics(
            distance_l1=0.4,
            changed_feature_count=1,
            lof_score=1.2,
        ),
    )
    engine = TestCounterfactualEngine()
    engine._normalized_lof_score = lambda _lof_score: 0.6

    score = engine._objective_score(
        candidate=candidate,
        preferences={},
        mutable_allowed=["BMI", "is_hypertension"],
        registry=registry,
    )

    expected = 0.50 * (0.4 / sqrt(2)) + 0.50 * 0.6
    assert abs(score - expected) < 1e-12


def test_proximity_uses_heom_for_mixed_feature_types() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
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
                actionable=True,
                default_mutable=True,
                global_min=0,
                global_max=2,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
            ),
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
        ],
    )
    engine = TestCounterfactualEngine()

    distance = engine._normalized_l1(
        candidate={"BMI": 25.0, "smoking_status": 1, "is_hypertension": 0},
        baseline={"BMI": 30.0, "smoking_status": 2, "is_hypertension": 1},
        registry=registry,
    )

    assert distance == sqrt((5.0 / 50.0) ** 2 + (1.0 / 2.0) ** 2 + 1.0)


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
    engine = TestCounterfactualEngine()
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
    engine = TestCounterfactualEngine()

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


def test_transition_ok_enforces_smoking_status_allowed_transitions() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="smoking_status",
                feature_type="ordinal",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=0,
                global_max=2,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=[],
                allowed_transitions={0: (0,), 1: (1,), 2: (1, 2)},
            )
        ],
    )
    engine = TestCounterfactualEngine()

    assert engine._transition_ok(
        candidate={"smoking_status": 1},
        baseline={"smoking_status": 2},
        mutable_allowed={"smoking_status"},
        registry=registry,
    )
    assert engine._transition_ok(
        candidate={"smoking_status": 2},
        baseline={"smoking_status": 2},
        mutable_allowed={"smoking_status"},
        registry=registry,
    )
    assert not engine._transition_ok(
        candidate={"smoking_status": 0},
        baseline={"smoking_status": 2},
        mutable_allowed={"smoking_status"},
        registry=registry,
    )
    assert not engine._transition_ok(
        candidate={"smoking_status": 0},
        baseline={"smoking_status": 1},
        mutable_allowed={"smoking_status"},
        registry=registry,
    )
    assert not engine._transition_ok(
        candidate={"smoking_status": 2},
        baseline={"smoking_status": 1},
        mutable_allowed={"smoking_status"},
        registry=registry,
    )


def test_infeasible_response_preserves_request_context_in_planner_input() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = TestCounterfactualEngine()
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
        },
    )()

    result = engine._process_candidates(
        request=req,
        prepared=prepared,
        raw_candidates=pd.DataFrame(),
        started=perf_counter(),
    )

    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
    assert result.input_prediction is not None
    assert result.planner_input.input_prediction is not None
    assert result.planner_input.input_prediction.class_name == "high_risk"
    assert result.planner_input.mutable_allowed == ["bmi"]


def test_process_candidates_surfaces_medical_only_infeasible_reason() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = TestCounterfactualEngine()
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
        },
    )()

    engine._coerce_candidate_features = lambda **_: {"age": 45, "bmi": 28.0, "glucose": 165}
    engine._immutable_ok = lambda *args, **kwargs: True
    engine._mutable_ok = lambda *args, **kwargs: True
    engine._directional_ok = lambda *args, **kwargs: True
    engine._transition_ok = lambda *args, **kwargs: True
    engine._medical_ok = lambda *args, **kwargs: False

    result = engine._process_candidates(
        request=req,
        prepared=prepared,
        raw_candidates=pd.DataFrame([{"age": 45, "bmi": 28.0, "glucose": 165.0}]),
        started=0.0,
    )

    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
    assert result.message == "Candidates exist but fail medical plausibility constraints."


def test_process_candidates_does_not_gate_on_immutable_mutable_or_lof_violations() -> None:
    req = CounterfactualRequest.model_validate(_request_payload(["bmi"]))
    engine = TestCounterfactualEngine()
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
        },
    )()
    engine.artifacts = object()  
    engine._as_model_input_df = lambda frame: frame  
    engine._predict_info = lambda _frame: PredictionInfo(  
        class_name="low_risk",
        probability_low_risk=0.8,
    )
    engine._lof_score = lambda _frame: 9.9  

    result = engine._process_candidates(
        request=req,
        prepared=prepared,
        raw_candidates=pd.DataFrame([{"age": 60, "bmi": 28.0, "glucose": 120.0}]),
        started=perf_counter(),
    )

    assert result.status == Status.FEASIBLE
    assert result.reason_code == ReasonCode.OK
    assert result.candidate is not None
    assert result.candidate.features["age"] == 60.0
    assert result.candidate.features["glucose"] == 120.0
    assert result.candidate.metrics.lof_score == 9.9
    assert result.validation.immutable_violation is False
    assert result.validation.mutable_violation is False
