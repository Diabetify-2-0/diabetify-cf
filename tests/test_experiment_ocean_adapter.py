from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
from experiments.engines.ocean_adapter import OceanCandidateGenerator, OceanSolverOptions


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


def test_ocean_solver_options_are_loaded_from_engine_config() -> None:
    options = OceanSolverOptions.from_config(
        {
            "engine_options": {
                "norm": 2,
                "attempt_count": 3,
                "seed_step": 101,
                "max_time_per_attempt_seconds": 4,
                "num_workers": 2,
            }
        }
    )

    assert options.norm == 2
    assert options.attempt_count == 3
    assert options.seed_step == 101
    assert options.max_time_per_attempt_seconds == 4
    assert options.num_workers == 2


def test_ocean_max_time_is_split_across_attempts() -> None:
    generator = OceanCandidateGenerator.__new__(OceanCandidateGenerator)
    generator.solver_options = OceanSolverOptions(attempt_count=3)
    request = SimpleNamespace(generation=SimpleNamespace(timeout_ms=10000))

    assert generator._max_time_per_attempt(request) == 4


def test_ocean_attempt_seed_is_deterministic() -> None:
    generator = OceanCandidateGenerator.__new__(OceanCandidateGenerator)
    generator.solver_options = OceanSolverOptions(seed_step=100)
    request = SimpleNamespace(generation=SimpleNamespace(random_seed=42))

    assert generator._attempt_seed(request, 0) == 42
    assert generator._attempt_seed(request, 2) == 242
