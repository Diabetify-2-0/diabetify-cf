from pathlib import Path

from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.schemas import CounterfactualRequest
from experiments.scripts.run_benchmark import (
    build_request_payload,
    engine_adapter_name,
    engine_output_label,
    load_effective_config,
    output_path_label,
)


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


def test_dice_engine_config_explicit_generation_defaults() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/dice.json"),
        scenario_config_path=Path("experiments/configs/scenarios/default_mutable.json"),
    )

    assert config["engine"] == "dice"
    assert config["engine_label"] == "dice"
    assert config["generation_method"] == "dice_random"
    assert config["total_cfs"] == 1
    assert config["timeout_ms"] == 5000


def test_nn_engine_config_explicit_search_options() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/nn.json"),
        scenario_config_path=Path("experiments/configs/scenarios/default_mutable.json"),
    )

    assert config["engine"] == "nn"
    assert config["engine_label"] == "nn"
    assert config["generation_method"] == "nearest_neighbor_projection"
    assert config["total_cfs"] == 3
    assert config["timeout_ms"] == 5000
    assert config["engine_options"] == {
        "candidate_pool_size": 256,
        "max_neighbors": 64,
        "max_changed_features": 3,
        "min_reference_low_risk_probability": None,
    }


def test_engine_output_label_can_distinguish_explicit_variant_labels() -> None:
    config = {
        "engine": "nn",
        "engine_label": "nn_variant_a",
    }

    assert engine_adapter_name(config) == "nn"
    assert engine_output_label(config) == "nn_variant_a"


def test_output_path_label_keeps_unknown_short_labels_readable() -> None:
    assert output_path_label("custom_engine") == "custom_engine"


def test_output_path_label_uses_known_engine_aliases() -> None:
    assert output_path_label("dice_constrained_native") == "dcn"
    assert output_path_label("nn_production") == "nnp"


def test_output_path_label_shortens_unknown_long_labels_deterministically() -> None:
    shortened = output_path_label("very_long_engine_variant_label", max_length=20)

    assert shortened.startswith("very_long_e")
    assert len(shortened) <= 20
    assert shortened == output_path_label("very_long_engine_variant_label", max_length=20)
