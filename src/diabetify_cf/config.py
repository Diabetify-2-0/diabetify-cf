from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_path_for(name: str) -> str:
    service_root = Path(__file__).resolve().parents[2]
    program_root = Path(__file__).resolve().parents[3]

    defaults = {
        "CF_MODEL_PATH": program_root / "diabetify-ml" / "xg_model.pkl",
        "CF_COLUMNS_PATH": program_root / "diabetify-ml" / "x_columns.pkl",
        "CF_REFERENCE_DATA_PATH": program_root / "diabetify-ml" / "shap_background.parquet",
        "CF_FEATURE_REGISTRY_PATH": service_root / "configs" / "feature_registry.json",
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
    prefetch_count: int = int(os.getenv("CF_PREFETCH_COUNT", "1"))

    default_total_cfs: int = int(os.getenv("CF_DEFAULT_TOTAL_CFS", "3"))
    request_timeout_ms: int = int(os.getenv("CF_REQUEST_TIMEOUT_MS", "5000"))
    max_lof_score: float = float(os.getenv("CF_MAX_LOF_SCORE", "2.5"))
    planner_enabled: bool = _bool_env("CF_PLANNER_ENABLED", True)
    planner_provider: str = os.getenv("CF_PLANNER_PROVIDER", "template")
    planner_intended_user: str = os.getenv("CF_PLANNER_INTENDED_USER", "clinician")
    planner_model: str = os.getenv("CF_PLANNER_MODEL", "gpt-4o-mini")
    planner_timeout_ms: int = int(os.getenv("CF_PLANNER_TIMEOUT_MS", "4000"))
    planner_temperature: float = float(os.getenv("CF_PLANNER_TEMPERATURE", "0.2"))
    planner_max_steps: int = int(os.getenv("CF_PLANNER_MAX_STEPS", "6"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_endpoint: str = os.getenv(
        "CF_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"
    )

    model_path: str = os.getenv("CF_MODEL_PATH", _default_path_for("CF_MODEL_PATH"))
    columns_path: str = os.getenv("CF_COLUMNS_PATH", _default_path_for("CF_COLUMNS_PATH"))
    preprocessor_path: str = os.getenv("CF_PREPROCESSOR_PATH", "")
    reference_data_path: str = os.getenv(
        "CF_REFERENCE_DATA_PATH", _default_path_for("CF_REFERENCE_DATA_PATH")
    )
    feature_registry_path: str = os.getenv(
        "CF_FEATURE_REGISTRY_PATH", _default_path_for("CF_FEATURE_REGISTRY_PATH")
    )
