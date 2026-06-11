import pytest

from diabetify_cf.config import Settings
from experiments.engines import (
    DiceExperimentAdapter,
    NearestNeighborExperimentAdapter,
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


def test_build_experiment_engine_returns_nn_adapter() -> None:
    adapter = build_experiment_engine(
        "nn",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
    )

    assert isinstance(adapter, NearestNeighborExperimentAdapter)


def test_build_experiment_engine_passes_config_to_nn_adapter() -> None:
    adapter = build_experiment_engine(
        "nn",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
        config={"engine_options": {"max_neighbors": 12, "max_changed_features": 2}},
    )

    assert isinstance(adapter, NearestNeighborExperimentAdapter)
    assert adapter.engine.options.max_neighbors == 12
    assert adapter.engine.options.max_changed_features == 2


def test_build_experiment_engine_rejects_unsupported_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment engine"):
        build_experiment_engine(
            "carla_face",
            settings=Settings(
                model_path="missing-model.pkl",
                columns_path="missing-columns.pkl",
            ),
        )
