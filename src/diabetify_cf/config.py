from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()
SERVICE_ROOT = Path(__file__).resolve().parents[2]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_or_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw


def _path_env_or_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((SERVICE_ROOT / path).resolve())


def _default_path_for(name: str) -> str:
    defaults = {
        "CF_MODEL_PATH": SERVICE_ROOT / "artifacts" / "models" / "xg_model.pkl",
        "CF_COLUMNS_PATH": SERVICE_ROOT / "artifacts" / "models" / "x_columns.pkl",
        "CF_REFERENCE_DATA_PATH": SERVICE_ROOT
        / "artifacts"
        / "reference"
        / "reference_data.parquet",
        "CF_FEATURE_REGISTRY_PATH": SERVICE_ROOT / "configs" / "feature_registry.json",
    }
    return str(defaults[name])


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("CF_LOG_LEVEL", "INFO")

    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://admin:password123@localhost:5672/")
    request_queue: str = os.getenv("CF_REQUEST_QUEUE", "ml.cf.request")
    response_queue: str = os.getenv("CF_RESPONSE_QUEUE", "ml.cf.response")

    max_rabbitmq_retries: int = int(os.getenv("CF_MAX_RABBITMQ_RETRIES", "5"))
    rabbitmq_retry_delay_sec: int = int(os.getenv("CF_RABBITMQ_RETRY_DELAY_SEC", "5"))
    rabbitmq_publish_retries: int = int(os.getenv("CF_RABBITMQ_PUBLISH_RETRIES", "3"))
    prefetch_count: int = int(os.getenv("CF_PREFETCH_COUNT", "1"))
    rabbitmq_enable_dlq: bool = _bool_env("CF_RABBITMQ_ENABLE_DLQ", False)
    rabbitmq_message_ttl_ms: int | None = (
        int(os.getenv("CF_RABBITMQ_MESSAGE_TTL_MS", ""))
        if os.getenv("CF_RABBITMQ_MESSAGE_TTL_MS")
        else None
    )
    request_dlq: str = os.getenv("CF_REQUEST_DLQ", "ml.cf.request.dlq")
    response_dlq: str = os.getenv("CF_RESPONSE_DLQ", "ml.cf.response.dlq")
    idempotency_cache_size: int = int(os.getenv("CF_IDEMPOTENCY_CACHE_SIZE", "1024"))

    max_lof_score: float = float(os.getenv("CF_MAX_LOF_SCORE", "1.5"))
    nn_candidate_pool_size: int = int(os.getenv("CF_NN_CANDIDATE_POOL_SIZE", "256"))
    nn_max_neighbors: int = int(os.getenv("CF_NN_MAX_NEIGHBORS", "64"))
    nn_min_reference_low_risk_probability: float | None = (
        float(os.getenv("CF_NN_MIN_REFERENCE_LOW_RISK_PROBABILITY", ""))
        if os.getenv("CF_NN_MIN_REFERENCE_LOW_RISK_PROBABILITY")
        else None
    )
    model_path: str = _path_env_or_default("CF_MODEL_PATH", _default_path_for("CF_MODEL_PATH"))
    columns_path: str = _path_env_or_default(
        "CF_COLUMNS_PATH", _default_path_for("CF_COLUMNS_PATH")
    )
    reference_data_path: str = _path_env_or_default(
        "CF_REFERENCE_DATA_PATH", _default_path_for("CF_REFERENCE_DATA_PATH")
    )
    feature_registry_path: str = _path_env_or_default(
        "CF_FEATURE_REGISTRY_PATH", _default_path_for("CF_FEATURE_REGISTRY_PATH")
    )

    def __post_init__(self) -> None:
        if self.app_env.strip().lower() not in {"prod", "production"}:
            return

        parsed_rabbitmq = urlparse(self.rabbitmq_url)
        if parsed_rabbitmq.username == "admin" and parsed_rabbitmq.password == "password123":
            raise ValueError("Production APP_ENV requires non-default RabbitMQ credentials.")
