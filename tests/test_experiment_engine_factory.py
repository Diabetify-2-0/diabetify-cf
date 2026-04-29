import pytest

from diabetify_cf.config import Settings
from experiments.engines import DiceExperimentAdapter, build_experiment_engine


def test_build_experiment_engine_returns_dice_adapter() -> None:
    adapter = build_experiment_engine(
        "dice",
        settings=Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        ),
    )

    assert isinstance(adapter, DiceExperimentAdapter)


def test_build_experiment_engine_rejects_unsupported_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment engine"):
        build_experiment_engine(
            "carla_face",
            settings=Settings(
                model_path="missing-model.pkl",
                columns_path="missing-columns.pkl",
            ),
        )
