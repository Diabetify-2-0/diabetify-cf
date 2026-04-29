from pathlib import Path

from diabetify_cf.config import _default_path_for, _env_or_default


def test_default_reference_data_path_uses_local_reference_artifact() -> None:
    path = Path(_default_path_for("CF_REFERENCE_DATA_PATH"))

    assert path.parts[-4:] == ("diabetify-cf", "artifacts", "reference", "reference_data.parquet")


def test_default_feature_registry_path_uses_configs_directory() -> None:
    path = Path(_default_path_for("CF_FEATURE_REGISTRY_PATH"))

    assert path.parts[-3:] == ("diabetify-cf", "configs", "feature_registry.json")


def test_default_model_artifacts_point_to_diabetify_ml_sibling_repo() -> None:
    model_path = Path(_default_path_for("CF_MODEL_PATH"))
    columns_path = Path(_default_path_for("CF_COLUMNS_PATH"))

    assert model_path.parts[-2:] == ("diabetify-ml", "xg_model.pkl")
    assert columns_path.parts[-2:] == ("diabetify-ml", "x_columns.pkl")


def test_empty_environment_value_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("CF_MODEL_PATH", "")

    assert _env_or_default("CF_MODEL_PATH", "fallback.pkl") == "fallback.pkl"
