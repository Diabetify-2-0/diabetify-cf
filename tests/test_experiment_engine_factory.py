import pytest

from diabetify_cf.config import Settings
from experiments.engines import (
    DiceExperimentAdapter,
    FeatureTweakExperimentAdapter,
    OceanExperimentAdapter,
    build_experiment_engine,
)


def test_build_experiment_engine_returns_dice_adapter() -> None:
    adapter = build_experiment_engine(
        "dice",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
    )

    assert isinstance(adapter, DiceExperimentAdapter)


def test_build_experiment_engine_returns_ocean_adapter() -> None:
    adapter = build_experiment_engine(
        "ocean",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
    )

    assert isinstance(adapter, OceanExperimentAdapter)


def test_build_experiment_engine_returns_ft_adapter() -> None:
    adapter = build_experiment_engine(
        "ft",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
    )

    assert isinstance(adapter, FeatureTweakExperimentAdapter)


def test_build_experiment_engine_passes_config_to_ocean_adapter() -> None:
    adapter = build_experiment_engine(
        "ocean",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
        config={"engine_options": {"attempt_count": 2, "norm": 1}},
    )

    assert isinstance(adapter, OceanExperimentAdapter)
    assert adapter.engine.solver_options.attempt_count == 2
    assert adapter.engine.solver_options.norm == 1


def test_build_experiment_engine_passes_config_to_ft_adapter() -> None:
    adapter = build_experiment_engine(
        "ft",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
        config={"engine_options": {"max_changed_features": 3, "beam_width": 8}},
    )

    assert isinstance(adapter, FeatureTweakExperimentAdapter)
    assert adapter.engine.options.max_changed_features == 3
    assert adapter.engine.options.beam_width == 8


def test_build_experiment_engine_rejects_unsupported_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment engine"):
        build_experiment_engine(
            "carla_face",
            settings=Settings(
                model_path="missing-model.pkl",
                columns_path="missing-columns.pkl",
            ),
        )
