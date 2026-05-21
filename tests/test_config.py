from pathlib import Path

import pytest

from diabetify_cf.config import (
    SERVICE_ROOT,
    Settings,
    _default_path_for,
    _env_or_default,
    _path_env_or_default,
)


def test_default_reference_data_path_uses_local_reference_artifact() -> None:
    path = Path(_default_path_for("CF_REFERENCE_DATA_PATH"))

    assert path.parts[-4:] == ("diabetify-cf", "artifacts", "reference", "reference_data.parquet")


def test_default_feature_registry_path_uses_configs_directory() -> None:
    path = Path(_default_path_for("CF_FEATURE_REGISTRY_PATH"))

    assert path.parts[-3:] == ("diabetify-cf", "configs", "feature_registry.json")


def test_default_model_artifacts_use_local_artifacts_directory() -> None:
    model_path = Path(_default_path_for("CF_MODEL_PATH"))
    columns_path = Path(_default_path_for("CF_COLUMNS_PATH"))

    assert model_path.parts[-4:] == ("diabetify-cf", "artifacts", "models", "xg_model.pkl")
    assert columns_path.parts[-4:] == ("diabetify-cf", "artifacts", "models", "x_columns.pkl")


def test_empty_environment_value_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("CF_MODEL_PATH", "")

    assert _env_or_default("CF_MODEL_PATH", "fallback.pkl") == "fallback.pkl"


def test_relative_artifact_path_is_resolved_from_repo_root(monkeypatch) -> None:
    monkeypatch.setenv("CF_MODEL_PATH", "artifacts/models/xg_model.pkl")

    resolved = Path(_path_env_or_default("CF_MODEL_PATH", "fallback.pkl"))

    assert resolved == (SERVICE_ROOT / "artifacts" / "models" / "xg_model.pkl").resolve()


def test_absolute_artifact_path_is_preserved(monkeypatch) -> None:
    absolute = SERVICE_ROOT / "artifacts" / "models" / "xg_model.pkl"
    monkeypatch.setenv("CF_MODEL_PATH", str(absolute))

    resolved = Path(_path_env_or_default("CF_MODEL_PATH", "fallback.pkl"))

    assert resolved == absolute


def test_prod_rejects_default_rabbitmq_credentials() -> None:
    with pytest.raises(ValueError, match="non-default RabbitMQ credentials"):
        Settings(
            app_env="prod",
            rabbitmq_url="amqp://admin:password123@localhost:5672/",
        )


def test_prod_accepts_non_default_rabbitmq_credentials() -> None:
    settings = Settings(
        app_env="prod",
        rabbitmq_url="amqp://cf_user:strong-password@rabbitmq:5672/",
        planner_provider="template",
    )

    assert settings.app_env == "prod"
