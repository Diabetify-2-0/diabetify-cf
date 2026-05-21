from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from diabetify_cf.config import Settings
from diabetify_cf.messaging.rabbitmq_service import RabbitMQCFService
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualResponse, ValidationSummary


def _settings() -> Settings:
    return replace(
        Settings(),
        rabbitmq_publish_retries=2,
        rabbitmq_retry_delay_sec=0,
        idempotency_cache_size=8,
    )


def _response(request_id: str) -> CounterfactualResponse:
    return CounterfactualResponse(
        request_id=request_id,
        status=Status.INFEASIBLE,
        reason_code=ReasonCode.NO_MUTABLE_FEATURE,
        message="No mutable feature selected by user.",
        model_version="xgb_v1",
        cf_engine_version="test_engine",
        validation=ValidationSummary(
            immutable_violation=False,
            mutable_compliance=True,
            medical_rules_passed=True,
        ),
    )


class CountingEngine:
    engine_version = "test_engine"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: Any) -> CounterfactualResponse:
        self.calls += 1
        return _response(request.request_id)


class FakeChannel:
    is_open = True

    def __init__(self, fail_publish_count: int = 0) -> None:
        self.fail_publish_count = fail_publish_count
        self.publish_count = 0
        self.acked: list[int] = []
        self.published_bodies: list[bytes] = []

    def basic_publish(self, **kwargs: Any) -> None:
        self.publish_count += 1
        if self.publish_count <= self.fail_publish_count:
            raise RuntimeError("transient publish failure")
        self.published_bodies.append(kwargs["body"])

    def basic_ack(self, delivery_tag: int) -> None:
        self.acked.append(delivery_tag)


class FakeMethod:
    delivery_tag = 42


class FakeProperties:
    reply_to = None
    correlation_id = "job-1"


def _request_payload() -> bytes:
    payload = {
        "request_id": "job-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "xgb_v1",
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "bmi": 31.2}},
        "constraints": {"mutable_allowed": []},
    }
    return json.dumps(payload).encode("utf-8")


def test_publish_response_retries_transient_failure() -> None:
    service = RabbitMQCFService(settings=_settings(), engine=CountingEngine())  # type: ignore[arg-type]
    channel = FakeChannel(fail_publish_count=1)
    service.channel = channel  # type: ignore[assignment]

    service._publish_response(
        response_queue="ml.cf.response",
        correlation_id="job-1",
        response=_response("job-1"),
    )

    assert channel.publish_count == 2
    assert len(channel.published_bodies) == 1


def test_on_message_uses_cached_response_for_duplicate_request_id() -> None:
    engine = CountingEngine()
    service = RabbitMQCFService(settings=_settings(), engine=engine)  # type: ignore[arg-type]
    channel = FakeChannel()
    service.channel = channel  # type: ignore[assignment]
    body = _request_payload()

    service._on_message(channel, FakeMethod(), FakeProperties(), body)  # type: ignore[arg-type]
    service._on_message(channel, FakeMethod(), FakeProperties(), body)  # type: ignore[arg-type]

    assert engine.calls == 1
    assert channel.acked == [42, 42]
    assert len(channel.published_bodies) == 2
    snapshot = service.get_health_snapshot()
    assert snapshot["processed_count"] == 2
    assert snapshot["reason_counts"] == {"NO_MUTABLE_FEATURE": 2}
    assert snapshot["last_request_id"] == "job-1"


def test_validation_error_response_is_sanitized() -> None:
    service = RabbitMQCFService(settings=_settings(), engine=CountingEngine())  # type: ignore[arg-type]
    channel = FakeChannel()
    service.channel = channel  # type: ignore[assignment]
    body = json.dumps({"request_id": "job-1"}).encode("utf-8")

    service._on_message(channel, FakeMethod(), FakeProperties(), body)  # type: ignore[arg-type]

    published = json.loads(channel.published_bodies[0].decode("utf-8"))
    assert published["status"] == "ERROR"
    assert published["reason_code"] == "INVALID_INPUT_SCHEMA"
    assert published["message"] == "Invalid counterfactual request payload."
    assert "errors" not in published["message"]
