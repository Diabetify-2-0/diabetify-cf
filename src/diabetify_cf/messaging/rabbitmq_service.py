from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pika
from pika.exceptions import AMQPConnectionError
from pydantic import ValidationError

from diabetify_cf.config import Settings
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    ValidationSummary,
)


class RabbitMQCFService:
    def __init__(self, settings: Settings, engine: CounterfactualEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.logger = logging.getLogger("diabetify_cf")
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.channel.Channel | None = None
        self.is_running = False

    def start(self) -> None:
        self._setup_connection()
        self.is_running = True

        assert self.channel is not None
        self.channel.basic_consume(
            queue=self.settings.request_queue,
            on_message_callback=self._on_message,
        )
        self.logger.info(
            "CF worker started: request_queue=%s response_queue=%s",
            self.settings.request_queue,
            self.settings.response_queue,
        )

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Interrupt received, shutting down service.")
            self.stop()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        if self.channel is not None and self.channel.is_open:
            self.channel.stop_consuming()
            self.channel.close()

        if self.connection is not None and self.connection.is_open:
            self.connection.close()

        self.logger.info("CF worker stopped.")

    def _setup_connection(self) -> None:
        for attempt in range(1, self.settings.max_rabbitmq_retries + 1):
            try:
                self.connection = pika.BlockingConnection(
                    pika.URLParameters(self.settings.rabbitmq_url)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.settings.request_queue, durable=True)
                self.channel.queue_declare(queue=self.settings.response_queue, durable=True)
                self.channel.basic_qos(prefetch_count=self.settings.prefetch_count)
                return
            except AMQPConnectionError as err:
                self.logger.warning(
                    "RabbitMQ connect attempt %d/%d failed: %s",
                    attempt,
                    self.settings.max_rabbitmq_retries,
                    err,
                )
                time.sleep(self.settings.rabbitmq_retry_delay_sec)

        raise RuntimeError("Failed to connect to RabbitMQ after maximum retries.")

    def _on_message(  # noqa: PLR0913
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        started = time.perf_counter()
        response_queue = properties.reply_to or self.settings.response_queue
        correlation_id = properties.correlation_id
        engine_version = getattr(self.engine, "engine_version", "unknown")

        try:
            payload = json.loads(body.decode("utf-8"))
            request = CounterfactualRequest.model_validate(payload)
            if not correlation_id:
                correlation_id = request.request_id

            response = self.engine.generate(request)
        except ValidationError as err:
            request_id = self._extract_request_id(body)
            response = CounterfactualResponse(
                request_id=request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA,
                message=f"Invalid request schema: {err.errors()}",
                model_version="unknown",
                cf_engine_version=engine_version,
                runtime_ms=int((time.perf_counter() - started) * 1000),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=False,
                    medical_rules_passed=False,
                ),
                timestamp=datetime.now(timezone.utc),
            )
            if not correlation_id:
                correlation_id = request_id
        except Exception as err:  # noqa: BLE001
            request_id = self._extract_request_id(body)
            response = CounterfactualResponse(
                request_id=request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INTERNAL_ERROR,
                message=f"Unhandled engine error: {err}",
                model_version="unknown",
                cf_engine_version=engine_version,
                runtime_ms=int((time.perf_counter() - started) * 1000),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=False,
                    medical_rules_passed=False,
                ),
                timestamp=datetime.now(timezone.utc),
            )
            if not correlation_id:
                correlation_id = request_id

        self._publish_response(
            response_queue=response_queue, correlation_id=correlation_id, response=response
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def _publish_response(
        self,
        response_queue: str,
        correlation_id: str | None,
        response: CounterfactualResponse,
    ) -> None:
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel is not initialized.")

        body = json.dumps(response.to_wire()).encode("utf-8")
        self.channel.basic_publish(
            exchange="",
            routing_key=response_queue,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                correlation_id=correlation_id,
                delivery_mode=2,
                timestamp=int(time.time()),
            ),
        )

    @staticmethod
    def _extract_request_id(body: bytes) -> str:
        try:
            payload: Any = json.loads(body.decode("utf-8"))
            request_id = payload.get("request_id")
            if isinstance(request_id, str) and request_id.strip():
                return request_id
        except Exception:  # noqa: BLE001
            pass
        return "unknown"
