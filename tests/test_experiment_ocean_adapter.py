from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
from experiments.engines.ocean_adapter import OceanCandidateGenerator


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


def test_ocean_request_bounds_lock_non_mutable_features() -> None:
    generator = OceanCandidateGenerator.__new__(OceanCandidateGenerator)
    generator.artifacts = SimpleNamespace(reference_data=pd.DataFrame({"BMI": [20.0, 30.0, 40.0]}))
    prepared = SimpleNamespace(
        instance_features={"BMI": 31.5},
        mutable_allowed=[],
        permitted_range={},
    )

    assert generator._request_bounds("BMI", _feature("BMI"), prepared) == (31.5, 31.5)


def test_ocean_request_bounds_use_prepared_permitted_range_for_mutable_features() -> None:
    generator = OceanCandidateGenerator.__new__(OceanCandidateGenerator)
    generator.artifacts = SimpleNamespace(reference_data=pd.DataFrame({"BMI": [20.0, 30.0, 40.0]}))
    prepared = SimpleNamespace(
        instance_features={"BMI": 31.5},
        mutable_allowed=["BMI"],
        permitted_range={"BMI": [25.0, 31.5]},
    )

    assert generator._request_bounds("BMI", _feature("BMI"), prepared) == (25.0, 31.5)


def test_ocean_discrete_levels_are_clipped_to_request_bounds() -> None:
    generator = OceanCandidateGenerator.__new__(OceanCandidateGenerator)
    generator.artifacts = SimpleNamespace(
        reference_data=pd.DataFrame({"activity": [0, 1, 2, 3, 4, 5]})
    )

    levels = generator._discrete_levels(
        "activity",
        _feature("activity", feature_type="ordinal"),
        lower=2.0,
        upper=4.0,
    )

    assert levels == [2.0, 3.0, 4.0]
