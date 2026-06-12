from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
from experiments.engines.nn_adapter import NearestNeighborCandidateGenerator, NearestNeighborOptions


def _feature(
    name: str,
    *,
    feature_type: str = "continuous",
    global_min: float = 0.0,
    global_max: float = 50.0,
    cost_weight: float = 1.0,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type=feature_type,
        immutable=False,
        actionable=True,
        default_mutable=True,
        global_min=global_min,
        global_max=global_max,
        cost_weight=cost_weight,
        preferred_direction="any",
        aliases=[],
    )


def test_nn_options_are_loaded_from_engine_config() -> None:
    options = NearestNeighborOptions.from_config(
        {
            "engine_options": {
                "candidate_pool_size": 32,
                "max_neighbors": 12,
                "max_changed_features": 2,
                "min_reference_low_risk_probability": 0.7,
            }
        }
    )

    assert options.candidate_pool_size == 32
    assert options.max_neighbors == 12
    assert options.max_changed_features == 2
    assert options.min_reference_low_risk_probability == 0.7


def test_nn_rank_changed_features_uses_weighted_normalized_distance() -> None:
    generator = NearestNeighborCandidateGenerator.__new__(NearestNeighborCandidateGenerator)
    prepared = SimpleNamespace(
        registry=SimpleNamespace(
            get=lambda name: {
                "BMI": _feature("BMI", global_min=10.0, global_max=60.0, cost_weight=1.0),
                "smoking_status": _feature(
                    "smoking_status",
                    feature_type="ordinal",
                    global_min=0.0,
                    global_max=2.0,
                    cost_weight=1.0,
                ),
            }.get(name)
        ),
        permitted_range={"BMI": [18.5, 35.0], "smoking_status": [0.0, 2.0]},
    )
    neighbor = pd.Series({"BMI": 29.0, "smoking_status": 1.0})
    baseline = {"BMI": 31.0, "smoking_status": 0.0}

    ranked = generator._rank_changed_features(
        neighbor=neighbor,
        baseline=baseline,
        mutable_allowed=["BMI", "smoking_status"],
        prepared=prepared,
    )

    assert ranked == ["BMI", "smoking_status"]


def test_nn_project_neighbors_keeps_immutable_features_from_baseline() -> None:
    generator = NearestNeighborCandidateGenerator.__new__(NearestNeighborCandidateGenerator)
    generator.options = NearestNeighborOptions(max_changed_features=2)
    generator.artifacts = SimpleNamespace(feature_columns=["age", "BMI", "smoking_status"])
    prepared = SimpleNamespace(
        registry=SimpleNamespace(
            get=lambda name: {
                "BMI": _feature("BMI", global_min=10.0, global_max=60.0),
                "smoking_status": _feature(
                    "smoking_status",
                    feature_type="ordinal",
                    global_min=0.0,
                    global_max=2.0,
                ),
            }.get(name)
        ),
        permitted_range={"BMI": [18.5, 35.0], "smoking_status": [0.0, 2.0]},
    )
    baseline = {"age": 45, "BMI": 31.0, "smoking_status": 0}
    neighbor = pd.Series({"age": 60, "BMI": 27.0, "smoking_status": 1})

    projected = generator._project_neighbors(
        ranked_neighbors=[neighbor],
        baseline=baseline,
        mutable_allowed=["BMI", "smoking_status"],
        prepared=prepared,
    )

    assert projected
    assert all(candidate["age"] == 45 for candidate in projected)
    assert any(candidate["BMI"] == 27.0 for candidate in projected)
