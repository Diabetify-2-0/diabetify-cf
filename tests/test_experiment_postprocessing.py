from time import perf_counter

import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest
from experiments.postprocessing import ExperimentPostprocessor


class _AlwaysLowRiskModel:
    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.8, 0.2] for _ in range(len(frame))]

    def predict(self, frame: pd.DataFrame) -> list[int]:
        return [0 for _ in range(len(frame))]


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
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=["bmi"],
            ),
        ],
    )


def _request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-postprocess",
            "instance": {"features": {"age": 45, "BMI": 31.2}},
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "constraints": {
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI"],
                "feature_bounds": {"BMI": {"min": 18.5, "max": 35.0}},
            },
        }
    )


def _postprocessor() -> ExperimentPostprocessor:
    artifacts = ModelArtifacts(
        model=_AlwaysLowRiskModel(),
        feature_columns=["age", "BMI"],
        reference_data=pd.DataFrame([{"age": 45, "BMI": 31.2}]),
        feature_registry=_registry(),
        lof_model=None,
    )
    return ExperimentPostprocessor(artifacts=artifacts, max_lof_score=2.5)


def test_postprocessor_rejects_candidates_outside_request_feature_bounds() -> None:
    postprocessor = _postprocessor()
    request = _request()
    prepared = postprocessor.prepare(request)

    result = postprocessor.process(
        request=request,
        prepared=prepared,
        raw_candidates=pd.DataFrame([{"age": 45, "BMI": 37.0}]),
        started=perf_counter(),
    )

    assert result.status == Status.INFEASIBLE
    assert result.reason_code == ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
    assert "feature bounds" in result.message
    assert result.candidates == []


def test_postprocessor_accepts_candidates_inside_request_feature_bounds() -> None:
    postprocessor = _postprocessor()
    request = _request()
    prepared = postprocessor.prepare(request)

    result = postprocessor.process(
        request=request,
        prepared=prepared,
        raw_candidates=pd.DataFrame([{"age": 45, "BMI": 29.0}]),
        started=perf_counter(),
    )

    assert result.status == Status.FEASIBLE
    assert result.reason_code == ReasonCode.OK
    assert result.candidates[0].delta == {"BMI": -2.2}
