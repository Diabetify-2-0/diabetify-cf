from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from diabetify_cf.verification.backend import HttpBackendCounterfactualGateway


AuthMode = Literal["bearer_token", "login"]
_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


@dataclass(frozen=True)
class LoginCredentials:
    email: str
    password: str


@dataclass(frozen=True)
class BootstrapUser:
    name: str
    gender: str | None = None
    dob: str | None = None


@dataclass(frozen=True)
class LauncherConfig:
    backend_base_url: str
    scenarios_path: str
    output_dir: str
    suites: tuple[str, ...] = ()
    auth_mode: AuthMode = "bearer_token"
    bearer_token: str | None = None
    login: LoginCredentials | None = None
    bootstrap_user: BootstrapUser | None = None
    register_if_missing: bool = False
    skip_health_check: bool = False
    health_timeout_seconds: float = 60.0
    health_poll_interval_seconds: float = 2.0
    poll_interval_seconds: float = 1.0
    poll_timeout_seconds: float = 300.0


def load_launcher_config(path: str | Path) -> LauncherConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launcher config must be a JSON object")
    payload = _resolve_env_placeholders(payload)

    auth_mode = str(payload.get("auth_mode", "bearer_token"))
    if auth_mode not in {"bearer_token", "login"}:
        raise ValueError(f"unsupported auth_mode: {auth_mode}")

    login_payload = payload.get("login")
    login: LoginCredentials | None = None
    if isinstance(login_payload, dict):
        email = str(login_payload.get("email", "")).strip()
        password = str(login_payload.get("password", "")).strip()
        if email and password:
            login = LoginCredentials(email=email, password=password)

    bootstrap_payload = payload.get("bootstrap_user")
    bootstrap_user: BootstrapUser | None = None
    if isinstance(bootstrap_payload, dict):
        name = str(bootstrap_payload.get("name", "")).strip()
        if name:
            bootstrap_user = BootstrapUser(
                name=name,
                gender=_optional_string(bootstrap_payload.get("gender")),
                dob=_optional_string(bootstrap_payload.get("dob")),
            )

    config = LauncherConfig(
        backend_base_url=str(payload["backend_base_url"]),
        scenarios_path=str(payload["scenarios_path"]),
        output_dir=str(payload["output_dir"]),
        suites=tuple(str(item) for item in payload.get("suites", [])),
        auth_mode=auth_mode,
        bearer_token=_optional_string(payload.get("bearer_token")),
        login=login,
        bootstrap_user=bootstrap_user,
        register_if_missing=bool(payload.get("register_if_missing", False)),
        skip_health_check=bool(payload.get("skip_health_check", False)),
        health_timeout_seconds=float(payload.get("health_timeout_seconds", 60.0)),
        health_poll_interval_seconds=float(payload.get("health_poll_interval_seconds", 2.0)),
        poll_interval_seconds=float(payload.get("poll_interval_seconds", 1.0)),
        poll_timeout_seconds=float(payload.get("poll_timeout_seconds", 300.0)),
    )
    _validate_launcher_config(config)
    return config


def resolve_backend_bearer_token(config: LauncherConfig) -> str | None:
    if config.auth_mode == "bearer_token":
        return config.bearer_token

    if config.login is None:
        raise ValueError("login auth_mode requires login.email and login.password")

    gateway = HttpBackendCounterfactualGateway(base_url=config.backend_base_url)
    try:
        payload = _login_with_gateway(gateway, config.login)
    except RuntimeError as err:
        if not _should_bootstrap_user(config, err):
            raise
        _register_bootstrap_user(gateway, config)
        payload = _login_with_gateway(gateway, config.login)

    token = payload.get("data")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("backend login did not return a bearer token string")
    return token


def _validate_launcher_config(config: LauncherConfig) -> None:
    if not config.backend_base_url.strip():
        raise ValueError("backend_base_url is required")
    if not config.scenarios_path.strip():
        raise ValueError("scenarios_path is required")
    if not config.output_dir.strip():
        raise ValueError("output_dir is required")
    if config.auth_mode == "bearer_token" and not config.bearer_token:
        raise ValueError("bearer_token auth_mode requires bearer_token")
    if config.auth_mode == "login" and config.login is None:
        raise ValueError("login auth_mode requires login.email and login.password")
    if config.register_if_missing and config.auth_mode != "login":
        raise ValueError("register_if_missing is only supported with login auth_mode")
    if config.register_if_missing and config.bootstrap_user is None:
        raise ValueError("register_if_missing requires bootstrap_user.name")


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _resolve_env_placeholders(value: object) -> object:
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    match = _ENV_PATTERN.match(value.strip())
    if not match:
        return value

    env_name = match.group(1)
    resolved = os.getenv(env_name)
    if resolved is None:
        raise ValueError(f"environment variable '{env_name}' is not set")
    return resolved


def _login_with_gateway(
    gateway: HttpBackendCounterfactualGateway,
    credentials: LoginCredentials,
) -> dict[str, Any]:
    payload = gateway._request_json(
        "POST",
        "/users/login",
        {
            "email": credentials.email,
            "password": credentials.password,
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError("backend login returned a non-object payload")
    return payload


def _should_bootstrap_user(config: LauncherConfig, err: RuntimeError) -> bool:
    return (
        config.register_if_missing
        and config.bootstrap_user is not None
        and "HTTP 404" in str(err)
        and "User not found" in str(err)
    )


def _register_bootstrap_user(
    gateway: HttpBackendCounterfactualGateway,
    config: LauncherConfig,
) -> None:
    if config.login is None or config.bootstrap_user is None:
        raise ValueError("bootstrap registration requires login credentials and bootstrap_user")

    payload: dict[str, object] = {
        "email": config.login.email,
        "password": config.login.password,
        "name": config.bootstrap_user.name,
    }
    if config.bootstrap_user.gender:
        payload["gender"] = config.bootstrap_user.gender
    if config.bootstrap_user.dob:
        payload["dob"] = config.bootstrap_user.dob

    gateway._request_json("POST", "/users/", payload)
