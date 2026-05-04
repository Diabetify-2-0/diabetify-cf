from pathlib import Path

from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.schemas import CounterfactualRequest
from experiments.scripts.run_benchmark import (
    build_request_payload,
    engine_adapter_name,
    engine_output_label,
    load_effective_config,
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


def test_dice_engine_config_explicit_generation_defaults() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/dice.json"),
        scenario_config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert config["engine"] == "dice"
    assert config["engine_label"] == "dice"
    assert config["generation_method"] == "dice_genetic"
    assert config["total_cfs"] == 3
    assert config["timeout_ms"] == 5000


def test_dace_engine_config_explicit_solver_options() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/dace.json"),
        scenario_config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert config["engine"] == "dace"
    assert config["engine_label"] == "dace"
    assert config["generation_method"] == "dace_rf_surrogate"
    assert config["total_cfs"] == 3
    assert config["timeout_ms"] == 15000
    assert config["engine_options"] == {
        "surrogate_n_estimators": 64,
        "surrogate_max_depth": 6,
        "max_changed_features": 3,
        "max_candidates_per_feature": 24,
        "threshold_epsilon": 0.0001,
        "solver": "PULP_CBC_CMD",
        "relative_gap": None,
    }


def test_ocean_engine_config_explicit_solver_options() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/ocean.json"),
        scenario_config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert config["engine"] == "ocean"
    assert config["engine_label"] == "ocean"
    assert config["generation_method"] == "ocean_cp"
    assert config["total_cfs"] == 1
    assert config["timeout_ms"] == 15000
    assert config["engine_options"] == {
        "norm": 1,
        "attempt_count": 2,
        "seed_step": 997,
        "max_time_per_attempt_seconds": None,
        "num_workers": None,
    }


def test_ft_engine_config_explicit_search_options() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/ft.json"),
        scenario_config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert config["engine"] == "ft"
    assert config["engine_label"] == "ft"
    assert config["generation_method"] == "feature_tweak_style"
    assert config["total_cfs"] == 3
    assert config["timeout_ms"] == 15000
    assert config["engine_options"] == {
        "max_changed_features": 2,
        "beam_width": 24,
        "max_candidates_to_evaluate": 300,
        "max_thresholds_per_feature": 16,
        "reference_values_per_feature": 16,
        "single_feature_grid_size": 25,
        "threshold_epsilon": 0.0001,
        "search_patience": 2,
    }


def test_engine_output_label_can_distinguish_same_adapter_variants() -> None:
    config = {
        "engine": "ocean",
        "engine_label": "ocean_attempt4",
    }

    assert engine_adapter_name(config) == "ocean"
    assert engine_output_label(config) == "ocean_attempt4"


def test_ocean_attempt4_config_uses_ocean_adapter_with_distinct_label() -> None:
    config = load_effective_config(
        engine_config_path=Path("experiments/configs/engines/ocean_attempt4.json"),
        scenario_config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert engine_adapter_name(config) == "ocean"
    assert engine_output_label(config) == "ocean_attempt4"
    assert config["timeout_ms"] == 30000
    assert config["engine_options"]["attempt_count"] == 4
