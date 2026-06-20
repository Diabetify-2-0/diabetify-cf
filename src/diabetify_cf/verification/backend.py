from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    PlannerInput,
    ValidationSummary,
)


class BackendCounterfactualGateway(ABC):
    @abstractmethod
    def get_health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_job_result(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class HttpBackendCounterfactualGateway(BackendCounterfactualGateway):
    base_url: str
    bearer_token: str | None = None
    timeout_seconds: float = 30.0

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/counterfactual/", payload)

    def get_health(self) -> dict[str, Any]:
        return self._request_json("GET", "/counterfactual/health")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/counterfactual/job/{job_id}/status")

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/counterfactual/job/{job_id}/result")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_base = self.base_url.rstrip("/")
        req = request.Request(
            normalized_base + path,
            method=method,
            headers=self._headers(),
        )
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req, data=body, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as err:
            message = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"backend returned HTTP {err.code}: {message}") from err
        except error.URLError as err:
            raise RuntimeError(f"backend request failed: {err}") from err

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("backend response must be a JSON object")
        return parsed

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers


class BackendCounterfactualEngineAdapter(CounterfactualEngine):
    def __init__(
        self,
        *,
        gateway: BackendCounterfactualGateway,
        poll_interval_seconds: float = 1.0,
        poll_timeout_seconds: float = 300.0,
    ) -> None:
        self.gateway = gateway
        self.poll_interval_seconds = max(0.1, poll_interval_seconds)
        self.poll_timeout_seconds = max(self.poll_interval_seconds, poll_timeout_seconds)

    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        submit_payload = _build_backend_submit_payload(request)
        submit_response = self.gateway.submit_job(submit_payload)
        job_id = _extract_job_id(submit_response)
        deadline = time.monotonic() + self.poll_timeout_seconds

        last_status_payload: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            status_payload = self.gateway.get_job_status(job_id)
            last_status_payload = status_payload
            job_status = _extract_job_status(status_payload)
            if job_status in {"completed", "infeasible", "failed", "cancelled"}:
                return self._resolve_terminal_response(
                    request=request,
                    job_id=job_id,
                    job_status=job_status,
                    status_payload=status_payload,
                )
            time.sleep(self.poll_interval_seconds)

        return _build_fallback_error_response(
            request=request,
            request_id=job_id,
            reason_code=ReasonCode.INTERNAL_ERROR,
            message="Backend polling exceeded timeout before terminal counterfactual result.",
        )

    def _resolve_terminal_response(
        self,
        *,
        request: CounterfactualRequest,
        job_id: str,
        job_status: str,
        status_payload: dict[str, Any],
    ) -> CounterfactualResponse:
        response = _load_backend_result_response(self.gateway, job_id)
        if response is not None:
            return response

        if job_status == "infeasible":
            return _build_fallback_error_response(
                request=request,
                request_id=job_id,
                reason_code=_extract_reason_code(status_payload, default=ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS),
                message=_extract_status_message(status_payload) or "Counterfactual job finished without feasible candidates.",
                status=Status.INFEASIBLE,
            )
        if job_status in {"failed", "cancelled"}:
            return _build_fallback_error_response(
                request=request,
                request_id=job_id,
                reason_code=_extract_reason_code(status_payload, default=ReasonCode.INTERNAL_ERROR),
                message=_extract_status_message(status_payload) or "Counterfactual backend job failed.",
                status=Status.ERROR,
            )

        return _build_fallback_error_response(
            request=request,
            request_id=job_id,
            reason_code=ReasonCode.INTERNAL_ERROR,
            message="Counterfactual backend returned a terminal status without result payload.",
        )


def wait_for_backend_health(
    gateway: BackendCounterfactualGateway,
    *,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    interval = max(0.0, poll_interval_seconds)
    last_payload: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        payload = gateway.get_health()
        last_payload = payload
        if _is_backend_healthy(payload):
            return payload
        if interval > 0:
            time.sleep(interval)

    details = _health_summary(last_payload)
    raise RuntimeError(f"backend health check did not become ready in time: {details}")


def _load_backend_result_response(
    gateway: BackendCounterfactualGateway,
    job_id: str,
) -> CounterfactualResponse | None:
    try:
        result_payload = gateway.get_job_result(job_id)
    except RuntimeError:
        return None
    return _extract_service_response(result_payload)


def _is_backend_healthy(payload: dict[str, Any]) -> bool:
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    service_status = data.get("service_status")
    if not isinstance(service_status, dict):
        return False
    running = service_status.get("running")
    rabbitmq_connected = service_status.get("rabbitmq_connected")
    return running is True and rabbitmq_connected is True


def _health_summary(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "no health payload received"
    data = payload.get("data")
    if not isinstance(data, dict):
        return "missing data.service_status"
    service_status = data.get("service_status")
    if not isinstance(service_status, dict):
        return "missing data.service_status"
    running = service_status.get("running")
    rabbitmq_connected = service_status.get("rabbitmq_connected")
    return f"running={running}, rabbitmq_connected={rabbitmq_connected}"


def _build_backend_submit_payload(counterfactual_request: CounterfactualRequest) -> dict[str, Any]:
    payload = counterfactual_request.model_dump(exclude_none=True)
    payload.pop("request_id", None)
    payload.pop("timestamp", None)
    return payload


def _extract_job_id(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("backend submit response missing data object")
    job_id = data.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RuntimeError("backend submit response missing job_id")
    return job_id


def _extract_job_status(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("backend status response missing data object")
    job_status = data.get("job_status")
    if not isinstance(job_status, str) or not job_status.strip():
        raise RuntimeError("backend status response missing job_status")
    return job_status.strip().lower()


def _extract_service_response(payload: dict[str, Any]) -> CounterfactualResponse | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, dict) or not result:
        return None
    return CounterfactualResponse.model_validate(result)


def _extract_reason_code(
    payload: dict[str, Any],
    *,
    default: ReasonCode,
) -> ReasonCode:
    data = payload.get("data")
    if not isinstance(data, dict):
        return default
    raw = data.get("reason_code")
    if not isinstance(raw, str) or not raw:
        return default
    try:
        return ReasonCode(raw)
    except ValueError:
        return default


def _extract_status_message(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    for field in ("error", "message"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_fallback_error_response(
    *,
    request: CounterfactualRequest,
    request_id: str,
    reason_code: ReasonCode,
    message: str,
    status: Status = Status.ERROR,
) -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id=request_id,
        status=status,
        reason_code=reason_code,
        message=message,
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_violation=False,
            medical_rules_passed=False,
        ),
        planner_input=PlannerInput(),
    )
