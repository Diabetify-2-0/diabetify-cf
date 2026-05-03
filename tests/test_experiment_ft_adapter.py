from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
from experiments.engines.ft_adapter import FeatureTweakCandidateGenerator, FeatureTweakOptions


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


def test_ft_options_are_loaded_from_engine_config() -> None:
    options = FeatureTweakOptions.from_config(
        {
            "engine_options": {
                "max_changed_features": 3,
                "beam_width": 12,
                "max_candidates_to_evaluate": 120,
                "max_thresholds_per_feature": 10,
                "reference_values_per_feature": 8,
                "single_feature_grid_size": 21,
                "threshold_epsilon": 0.001,
                "search_patience": 4,
            }
        }
    )

    assert options.max_changed_features == 3
    assert options.beam_width == 12
    assert options.max_candidates_to_evaluate == 120
    assert options.max_thresholds_per_feature == 10
    assert options.reference_values_per_feature == 8
    assert options.single_feature_grid_size == 21
    assert options.threshold_epsilon == 0.001
    assert options.search_patience == 4


def test_ft_bounds_use_prepared_permitted_range_for_mutable_features() -> None:
    generator = FeatureTweakCandidateGenerator.__new__(FeatureTweakCandidateGenerator)
    generator.artifacts = SimpleNamespace(reference_data=pd.DataFrame({"BMI": [20.0, 30.0, 40.0]}))
    prepared = SimpleNamespace(
        instance_features={"BMI": 31.5},
        mutable_allowed=["BMI"],
        permitted_range={"BMI": [25.0, 31.5]},
    )

    assert generator._bounds_for_feature("BMI", _feature("BMI"), prepared) == (25.0, 31.5)


def test_ft_continuous_candidate_values_use_threshold_sides() -> None:
    generator = FeatureTweakCandidateGenerator.__new__(FeatureTweakCandidateGenerator)
    generator.options = FeatureTweakOptions(max_thresholds_per_feature=10, threshold_epsilon=0.1)
    generator._feature_thresholds = {"BMI": [24.0, 28.0, 32.0]}

    values = generator._continuous_candidate_values(
        feature_name="BMI",
        baseline_value=30.0,
        lower=20.0,
        upper=35.0,
        exhaustive=False,
    )

    assert 27.9 in values
    assert 32.1 in values
    assert 20.0 in values
    assert 35.0 in values


def test_ft_ordinal_candidate_values_clip_to_bounds() -> None:
    generator = FeatureTweakCandidateGenerator.__new__(FeatureTweakCandidateGenerator)
    generator._feature_thresholds = {"activity": [1.2, 3.7, 5.9]}

    values = generator._ordinal_candidate_values(
        feature_name="activity",
        baseline_value=2.0,
        lower=2.0,
        upper=5.0,
    )

    assert values == [2.0, 3.0, 4.0, 5.0]


def test_ft_single_feature_search_uses_reference_values_and_grid() -> None:
    generator = FeatureTweakCandidateGenerator.__new__(FeatureTweakCandidateGenerator)
    generator.options = FeatureTweakOptions(
        reference_values_per_feature=4,
        single_feature_grid_size=5,
        max_candidates_to_evaluate=50,
    )
    generator._feature_thresholds = {"BMI": [24.0, 28.0]}
    generator.artifacts = SimpleNamespace(
        reference_data=pd.DataFrame({"BMI": [19.0, 21.0, 23.0, 25.0, 27.0, 29.0]}),
    )

    values = generator._candidate_values_for_feature(
        feature_name="BMI",
        feature=_feature("BMI"),
        baseline_value=26.0,
        lower=18.5,
        upper=30.0,
        exhaustive=True,
    )

    assert 18.5 in values
    assert 30.0 in values
    assert 21.0 in values
    assert 29.0 in values
