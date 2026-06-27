from __future__ import annotations

import os
from unittest.mock import patch

from diabetify_cf.verification.launcher import (
    BootstrapUser,
    LauncherConfig,
    LoginCredentials,
    load_launcher_config,
    resolve_backend_bearer_token,
)


def test_load_launcher_config_reads_login_mode_example() -> None:
    previous_base_url = os.environ.get("DIABETIFY_BACKEND_BASE_URL")
    previous_email = os.environ.get("DIABETIFY_TEST_USER_EMAIL")
    previous_password = os.environ.get("DIABETIFY_TEST_USER_PASSWORD")
    os.environ["DIABETIFY_BACKEND_BASE_URL"] = "http://localhost:8080"
    os.environ["DIABETIFY_TEST_USER_EMAIL"] = "tester@example.com"
    os.environ["DIABETIFY_TEST_USER_PASSWORD"] = "test-password"

    config = load_launcher_config(
        "evaluation/launcher/backend_suite_launcher.example.json"
    )

    try:
        assert config.backend_base_url == "http://localhost:8080"
        assert config.scenarios_path == "evaluation/fixtures"
        assert config.output_dir == "evaluation/reports/backend_suites"
        assert config.suites == (
            "actionability_core",
            "feasible_core",
            "infeasible_core",
            "repeatability_core",
        )
        assert config.auth_mode == "login"
        assert config.login is not None
        assert config.login.email == "tester@example.com"
        assert config.login.password == "test-password"
        assert config.register_if_missing is True
        assert config.bootstrap_user is not None
        assert config.bootstrap_user.name == "Counterfactual E2E"
        assert config.bootstrap_user.gender == "male"
        assert config.bootstrap_user.dob == "1990-01-01"
    finally:
        _restore_env("DIABETIFY_BACKEND_BASE_URL", previous_base_url)
        _restore_env("DIABETIFY_TEST_USER_EMAIL", previous_email)
        _restore_env("DIABETIFY_TEST_USER_PASSWORD", previous_password)


def test_resolve_backend_bearer_token_returns_static_token_for_bearer_mode() -> None:
    config = LauncherConfig(
        backend_base_url="http://localhost:8080",
        scenarios_path="evaluation/fixtures",
        output_dir="reports",
        auth_mode="bearer_token",
        bearer_token="static-token",
    )

    token = resolve_backend_bearer_token(config)

    assert token == "static-token"


def test_load_launcher_config_rejects_missing_bearer_token() -> None:
    try:
        load_launcher_config("evaluation/launcher/backend_suite_launcher.invalid_bearer.json")
    except ValueError as err:
        assert "requires bearer_token" in str(err)
    else:
        raise AssertionError("expected invalid launcher config to raise")


def test_load_launcher_config_rejects_missing_env_placeholder() -> None:
    previous_base_url = os.environ.get("DIABETIFY_BACKEND_BASE_URL")
    previous_email = os.environ.get("DIABETIFY_TEST_USER_EMAIL")
    previous_password = os.environ.get("DIABETIFY_TEST_USER_PASSWORD")
    os.environ.pop("DIABETIFY_BACKEND_BASE_URL", None)
    os.environ.pop("DIABETIFY_TEST_USER_EMAIL", None)
    os.environ.pop("DIABETIFY_TEST_USER_PASSWORD", None)

    try:
        load_launcher_config("evaluation/launcher/backend_suite_launcher.example.json")
    except ValueError as err:
        assert "environment variable 'DIABETIFY_BACKEND_BASE_URL' is not set" in str(err)
    else:
        raise AssertionError("expected missing env placeholder to raise")
    finally:
        _restore_env("DIABETIFY_BACKEND_BASE_URL", previous_base_url)
        _restore_env("DIABETIFY_TEST_USER_EMAIL", previous_email)
        _restore_env("DIABETIFY_TEST_USER_PASSWORD", previous_password)


def test_login_mode_requires_credentials() -> None:
    try:
        resolve_backend_bearer_token(
            LauncherConfig(
                backend_base_url="http://localhost:8080",
                scenarios_path="evaluation/fixtures",
                output_dir="reports",
                auth_mode="login",
                login=None,
            )
        )
    except ValueError as err:
        assert "requires login.email and login.password" in str(err)
    else:
        raise AssertionError("expected login mode without credentials to raise")


def test_login_mode_bootstraps_user_when_missing() -> None:
    config = LauncherConfig(
        backend_base_url="http://localhost:8080",
        scenarios_path="evaluation/fixtures",
        output_dir="reports",
        auth_mode="login",
        login=LoginCredentials(email="tester@example.com", password="secret123"),
        register_if_missing=True,
        bootstrap_user=BootstrapUser(
            name="Counterfactual E2E",
            gender="male",
            dob="1990-01-01",
        ),
    )

    calls: list[tuple[str, str, dict[str, str]]] = []

    def fake_request_json(
        self: object,
        method: str,
        path: str,
        payload: dict[str, str],
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        if path == "/users/login" and len(calls) == 1:
            raise RuntimeError(
                'backend returned HTTP 404: {"error":"No user associated with this email","message":"User not found","status":"error"}'
            )
        if path == "/users/":
            return {"status": "success"}
        if path == "/users/login":
            return {"data": "jwt-token"}
        raise AssertionError(f"unexpected request path: {path}")

    with patch(
        "diabetify_cf.verification.backend.HttpBackendCounterfactualGateway._request_json",
        new=fake_request_json,
    ):
        token = resolve_backend_bearer_token(config)

    assert token == "jwt-token"
    assert calls == [
        (
            "POST",
            "/users/login",
            {"email": "tester@example.com", "password": "secret123"},
        ),
        (
            "POST",
            "/users/",
            {
                "email": "tester@example.com",
                "password": "secret123",
                "name": "Counterfactual E2E",
                "gender": "male",
                "dob": "1990-01-01",
            },
        ),
        (
            "POST",
            "/users/login",
            {"email": "tester@example.com", "password": "secret123"},
        ),
    ]


def test_register_if_missing_requires_bootstrap_user() -> None:
    try:
        load_launcher_config("evaluation/launcher/backend_suite_launcher.invalid_register_if_missing.json")
    except ValueError as err:
        assert "requires bootstrap_user.name" in str(err)
    else:
        raise AssertionError("expected missing bootstrap user to raise")


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
