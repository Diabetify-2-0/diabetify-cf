from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "BackendCounterfactualEngineAdapter": "diabetify_cf.verification.backend",
    "BackendCounterfactualGateway": "diabetify_cf.verification.backend",
    "HttpBackendCounterfactualGateway": "diabetify_cf.verification.backend",
    "wait_for_backend_health": "diabetify_cf.verification.backend",
    "ExternalCounterfactualVerifier": "diabetify_cf.verification.external",
    "VerificationCandidateResult": "diabetify_cf.verification.external",
    "VerificationReport": "diabetify_cf.verification.external",
    "load_verification_scenarios": "diabetify_cf.verification.fixtures",
    "LauncherConfig": "diabetify_cf.verification.launcher",
    "LoginCredentials": "diabetify_cf.verification.launcher",
    "load_launcher_config": "diabetify_cf.verification.launcher",
    "resolve_backend_bearer_token": "diabetify_cf.verification.launcher",
    "build_report_payload": "diabetify_cf.verification.reporting",
    "write_report_json": "diabetify_cf.verification.reporting",
    "MetricSummary": "diabetify_cf.verification.runner",
    "ScenarioAggregate": "diabetify_cf.verification.runner",
    "ScenarioExpectation": "diabetify_cf.verification.runner",
    "ScenarioRunRecord": "diabetify_cf.verification.runner",
    "ScenarioRunner": "diabetify_cf.verification.runner",
    "VerificationScenario": "diabetify_cf.verification.runner",
    "DEFAULT_BACKEND_SUITES": "diabetify_cf.verification.suites",
    "VerificationSuite": "diabetify_cf.verification.suites",
    "build_backend_suite_index": "diabetify_cf.verification.suites",
    "build_suite_payload": "diabetify_cf.verification.suites",
    "load_suite_scenarios": "diabetify_cf.verification.suites",
    "select_verification_suites": "diabetify_cf.verification.suites",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
