from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.schemas import CounterfactualRequest
from experiments.scripts.run_benchmark import build_request_payload


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
            FeatureDefinition(
                name="is_bloodline",
                feature_type="binary",
                immutable=True,
                actionable=False,
                default_mutable=False,
                global_min=0,
                global_max=1,
                cost_weight=10.0,
                preferred_direction="any",
                aliases=[],
            ),
        ],
    )


def test_build_request_payload_uses_registry_defaults() -> None:
    payload = build_request_payload(
        row={"age": 45, "BMI": 31.2, "is_bloodline": 1},
        index=1,
        feature_columns=["age", "BMI", "is_bloodline"],
        registry=_registry(),
        config={},
    )

    request = CounterfactualRequest.model_validate(payload)

    assert request.constraints.immutable_features == ["age", "is_bloodline"]
    assert request.constraints.mutable_allowed == ["BMI"]
    assert request.constraints.feature_bounds["BMI"].min == 10
    assert request.constraints.feature_bounds["BMI"].max == 60


def test_build_request_payload_respects_explicit_mutable_allowed() -> None:
    payload = build_request_payload(
        row={"age": 45, "BMI": 31.2, "is_bloodline": 1},
        index=1,
        feature_columns=["age", "BMI", "is_bloodline"],
        registry=_registry(),
        config={"mutable_allowed": ["BMI"], "immutable_features": ["age"]},
    )

    request = CounterfactualRequest.model_validate(payload)

    assert request.constraints.immutable_features == ["age"]
    assert request.constraints.mutable_allowed == ["BMI"]


def test_build_request_payload_can_disable_default_mutable_allowed() -> None:
    payload = build_request_payload(
        row={"age": 45, "BMI": 31.2, "is_bloodline": 1},
        index=1,
        feature_columns=["age", "BMI", "is_bloodline"],
        registry=_registry(),
        config={"mutable_allowed": [], "use_default_mutable": False},
    )

    request = CounterfactualRequest.model_validate(payload)

    assert request.constraints.mutable_allowed == []
