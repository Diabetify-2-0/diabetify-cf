from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    PlannerInput,
    PredictionInfo,
    ValidationSummary,
)
from diabetify_cf.verification.backend import (
    BackendCounterfactualEngineAdapter,
    BackendCounterfactualGateway,
    HttpBackendCounterfactualGateway,
    _build_backend_submit_payload,
    wait_for_backend_health,
)


def _request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-backend",
            "model_version": "xgb_v1",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {"features": {"age": 45, "BMI": 31.2, "smoking_status": 2}},
            "constraints": {
                "immutable_features": ["age"],
                "mutable_allowed": ["BMI", "smoking_status"],
                "must_not_change": [],
            },
            "generation": {
                "total_cfs": 1,
                "method": "nearest_neighbor_projection",
                "random_seed": 42,
                "timeout_ms": 5000,
            },
        }
    )


def _service_response() -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id="job-123",
        status=Status.FEASIBLE,
        reason_code=ReasonCode.OK,
        message="Generated 1 feasible counterfactual candidate(s).",
        model_version="xgb_v1",
        cf_engine_version="nn_engine_v1",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_compliance=True,
            medical_rules_passed=True,
        ),
        candidates=[
            CounterfactualCandidate(
                candidate_id="cf_1",
                features={"age": 45, "BMI": 27.0, "smoking_status": 2},
                delta={"BMI": -4.2},
                prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.8),
                metrics=CandidateMetrics(
                    distance_l1=0.1,
                    changed_feature_count=1,
                    lof_score=1.1,
                    constraint_violations=0,
                ),
            )
        ],
        planner_input=PlannerInput(),
    )


@dataclass
class FakeGateway(BackendCounterfactualGateway):
    submit_payload: dict[str, Any] | None = None
    return_result_payload: bool = True

    def get_health(self) -> dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "service_status": {
                    "running": True,
                    "rabbitmq_connected": True,
                }
            },
        }

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.submit_payload = payload
        return {
            "status": "success",
            "data": {
                "job_id": "job-123",
                "status": "pending",
            },
        }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "job_id": job_id,
                "job_status": "completed",
                "reason_code": "OK",
            },
        }

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        if not self.return_result_payload:
            raise RuntimeError("backend returned HTTP 404: Job has no result payload")
        return {
            "status": "success",
            "data": {
                "job_id": job_id,
                "job_status": "completed",
                "reason_code": "OK",
                "result": _service_response().to_wire(),
            },
        }


def test_build_backend_submit_payload_strips_service_only_fields() -> None:
    payload = _build_backend_submit_payload(_request())

    assert "request_id" not in payload
    assert "timestamp" not in payload
    assert payload["model_version"] == "xgb_v1"
    assert payload["generation"]["method"] == "nearest_neighbor_projection"


def test_backend_counterfactual_engine_adapter_returns_service_payload() -> None:
    gateway = FakeGateway()
    adapter = BackendCounterfactualEngineAdapter(
        gateway=gateway,
        poll_interval_seconds=0.1,
        poll_timeout_seconds=1.0,
    )

    response = adapter.generate(_request())

    assert gateway.submit_payload is not None
    assert "request_id" not in gateway.submit_payload
    assert response.status == Status.FEASIBLE
    assert response.reason_code == ReasonCode.OK
    assert len(response.candidates) == 1


def test_backend_counterfactual_engine_adapter_falls_back_for_infeasible_job_without_result() -> None:
    @dataclass
    class InfeasibleGateway(BackendCounterfactualGateway):
        submit_payload: dict[str, Any] | None = None

        def get_health(self) -> dict[str, Any]:
            return {
                "status": "success",
                "data": {
                    "service_status": {
                        "running": True,
                        "rabbitmq_connected": True,
                    }
                },
            }

        def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.submit_payload = payload
            return {"status": "success", "data": {"job_id": "job-456", "status": "pending"}}

        def get_job_status(self, job_id: str) -> dict[str, Any]:
            return {
                "status": "success",
                "data": {
                    "job_id": job_id,
                    "job_status": "infeasible",
                    "reason_code": "NO_MUTABLE_FEATURE",
                    "message": "No mutable features were provided.",
                },
            }

        def get_job_result(self, job_id: str) -> dict[str, Any]:
            raise RuntimeError("backend returned HTTP 404: Job has no result payload")

    adapter = BackendCounterfactualEngineAdapter(
        gateway=InfeasibleGateway(),
        poll_interval_seconds=0.1,
        poll_timeout_seconds=1.0,
    )

    response = adapter.generate(_request())

    assert response.status == Status.INFEASIBLE
    assert response.reason_code == ReasonCode.NO_MUTABLE_FEATURE
    assert response.message == "No mutable features were provided."
    assert response.candidates == []


def test_http_backend_gateway_adds_bearer_token_header() -> None:
    gateway = HttpBackendCounterfactualGateway(
        base_url="http://localhost:8080",
        bearer_token="secret-token",
    )

    headers = gateway._headers()

    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer secret-token"


def test_wait_for_backend_health_returns_when_ready() -> None:
    gateway = FakeGateway()

    payload = wait_for_backend_health(
        gateway,
        timeout_seconds=0.2,
        poll_interval_seconds=0.0,
    )

    assert payload["data"]["service_status"]["running"] is True
    assert payload["data"]["service_status"]["rabbitmq_connected"] is True


def test_wait_for_backend_health_raises_when_backend_stays_unhealthy() -> None:
    @dataclass
    class UnhealthyGateway(BackendCounterfactualGateway):
        attempts: int = 0

        def get_health(self) -> dict[str, Any]:
            self.attempts += 1
            return {
                "status": "success",
                "data": {
                    "service_status": {
                        "running": False,
                        "rabbitmq_connected": False,
                    }
                },
            }

        def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("submit_job should not be called")

        def get_job_status(self, job_id: str) -> dict[str, Any]:
            raise AssertionError("get_job_status should not be called")

        def get_job_result(self, job_id: str) -> dict[str, Any]:
            raise AssertionError("get_job_result should not be called")

    gateway = UnhealthyGateway()

    try:
        wait_for_backend_health(
            gateway,
            timeout_seconds=0.05,
            poll_interval_seconds=0.0,
        )
    except RuntimeError as err:
        assert "running=False" in str(err)
        assert "rabbitmq_connected=False" in str(err)
    else:
        raise AssertionError("expected wait_for_backend_health to raise for unhealthy backend")
