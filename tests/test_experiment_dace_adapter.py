from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
from experiments.engines.dace_adapter import DaceCandidateGenerator, DaceSolverOptions


def _feature(name: str, feature_type: str = "continuous") -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type=feature_type,
        immutable=False,
        actionable=True,
        default_mutable=True,
        global_min=0.0,
        global_max=10.0,
        cost_weight=1.0,
        preferred_direction="any",
        aliases=[],
    )


def test_dace_options_are_loaded_from_engine_config() -> None:
    options = DaceSolverOptions.from_config(
        {
            "engine_options": {
                "surrogate_n_estimators": 32,
                "surrogate_max_depth": 5,
                "max_changed_features": 2,
                "max_candidates_per_feature": 10,
                "threshold_epsilon": 0.001,
                "solver": "PULP_CBC_CMD",
                "relative_gap": 0.05,
            }
        }
    )

    assert options.surrogate_n_estimators == 32
    assert options.surrogate_max_depth == 5
    assert options.max_changed_features == 2
    assert options.max_candidates_per_feature == 10
    assert options.threshold_epsilon == 0.001
    assert options.solver == "PULP_CBC_CMD"
    assert options.relative_gap == 0.05


def test_dace_feature_action_values_include_baseline_thresholds_and_reference_values() -> None:
    generator = DaceCandidateGenerator.__new__(DaceCandidateGenerator)
    generator.options = DaceSolverOptions(max_candidates_per_feature=8, threshold_epsilon=0.1)
    generator._surrogate_forest = SimpleNamespace()
    generator.artifacts = SimpleNamespace(
        reference_data=pd.DataFrame({"BMI": [19.0, 21.0, 23.0, 25.0, 27.0, 29.0]}),
    )
    generator._feature_thresholds = lambda feature_name, lower, upper: [24.0, 28.0]
    generator._reference_values = lambda feature_name, lower, upper: [21.0, 25.0, 29.0]

    values = generator._feature_action_values(
        feature_name="BMI",
        feature=_feature("BMI"),
        baseline=26.0,
        lower=18.5,
        upper=30.0,
    )

    assert 26.0 in values
    assert 23.9 in values
    assert 28.1 in values
    assert 25.0 in values
    assert 29.0 in values


def test_dace_bounds_use_permitted_range() -> None:
    generator = DaceCandidateGenerator.__new__(DaceCandidateGenerator)
    generator.artifacts = SimpleNamespace(reference_data=pd.DataFrame({"BMI": [20.0, 30.0, 40.0]}))
    prepared = SimpleNamespace(
        permitted_range={"BMI": [25.0, 31.5]},
    )

    assert generator._bounds_for_feature("BMI", _feature("BMI"), prepared) == (25.0, 31.5)
