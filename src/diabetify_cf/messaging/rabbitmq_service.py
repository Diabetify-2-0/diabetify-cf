"""RabbitMQ worker for the counterfactual service.

This module is the transport layer between the outside system and the
counterfactual engine. Its responsibilities are:
- connect to RabbitMQ,
- consume request messages from the request queue,
- validate and translate the payload into application models,
- call the engine,
- publish a structured response message back to RabbitMQ.

The worker intentionally catches validation and runtime failures so callers
receive a response payload with status and reason code instead of silence.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, OrderedDict
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
    """Blocking RabbitMQ consumer/publisher for counterfactual requests."""

    def __init__(self, settings: Settings, engine: CounterfactualEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.logger = logging.getLogger("diabetify_cf")
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.channel.Channel | None = None
        self.is_running = False
        self._response_cache: OrderedDict[str, CounterfactualResponse] = OrderedDict()
        self._status_counts: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._processed_count = 0
        self._last_request_id: str | None = None
        self._last_runtime_ms: int | None = None

    def start(self) -> None:
        """Open RabbitMQ resources and start the blocking consume loop."""
        self._setup_connection()
        self.is_running = True

        assert self.channel is not None
        # Register one callback that will be invoked for every request message
        # consumed from the request queue.
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
        """Gracefully stop consuming and close RabbitMQ resources."""
        if not self.is_running:
            return
        self.is_running = False

        # Stop the consumer before closing the channel so the blocking consume
        # loop can exit cleanly.
        if self.channel is not None and self.channel.is_open:
            self.channel.stop_consuming()
            self.channel.close()

        if self.connection is not None and self.connection.is_open:
            self.connection.close()

        self.logger.info("CF worker stopped.")

    def _setup_connection(self) -> None:
        """Create the RabbitMQ connection, channel, queues, and QoS policy."""
        for attempt in range(1, self.settings.max_rabbitmq_retries + 1):
            try:
                self.connection = pika.BlockingConnection(
                    pika.URLParameters(self.settings.rabbitmq_url)
                )
                self.channel = self.connection.channel()
                self._enable_publish_confirms()
                # Durable queues survive broker restarts. This matches the
                # worker role where requests and responses should not vanish
                # simply because RabbitMQ restarted.
                self._declare_queue_with_policy(
                    queue_name=self.settings.request_queue,
                    dlq_name=self.settings.request_dlq,
                )
                self._declare_queue_with_policy(
                    queue_name=self.settings.response_queue,
                    dlq_name=self.settings.response_dlq,
                )
                # Prefetch controls how many unacked messages this worker can
                # hold at once. The default config keeps processing mostly
                # one-request-at-a-time.
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

    def _enable_publish_confirms(self) -> None:
        if self.channel is None:
            return
        try:
            self.channel.confirm_delivery()
        except Exception as err:
            self.logger.warning("RabbitMQ publisher confirms unavailable: %s", err)

    def _declare_queue_with_policy(self, *, queue_name: str, dlq_name: str) -> None:
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel is not initialized.")

        arguments: dict[str, object] | None = None
        if self.settings.rabbitmq_enable_dlq:
            self.channel.queue_declare(queue=dlq_name, durable=True)
            arguments = {
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": dlq_name,
            }
            if self.settings.rabbitmq_message_ttl_ms is not None:
                arguments["x-message-ttl"] = self.settings.rabbitmq_message_ttl_ms

        self.channel.queue_declare(queue=queue_name, durable=True, arguments=arguments)

    def _on_message(
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        """Handle one incoming RabbitMQ message end-to-end.

        RabbitMQ passes four important pieces of information here:
        - `channel`: the active channel used to ack the message,
        - `method`: delivery metadata including delivery_tag,
        - `properties`: user metadata such as correlation_id and reply_to,
        - `body`: the raw request payload bytes.
        """
        started = time.perf_counter()
        # Prefer RPC-style reply_to when provided; otherwise publish to the
        # service's standard response queue.
        response_queue = properties.reply_to or self.settings.response_queue
        correlation_id = properties.correlation_id
        engine_version = getattr(self.engine, "engine_version", "unknown")

        try:
            # Decode JSON and validate it against the Pydantic schema before it
            # reaches the engine.
            payload = json.loads(body.decode("utf-8"))
            request = CounterfactualRequest.model_validate(payload)
            if not correlation_id:
                # If the sender omitted correlation_id, use request_id so the
                # caller still has a stable identifier to match response to
                # request.
                correlation_id = request.request_id

            cached_response = self._response_cache.get(request.request_id)
            if cached_response is not None:
                response = cached_response
                self._response_cache.move_to_end(request.request_id)
            else:
                response = self.engine.generate(request)
                self._cache_response(request.request_id, response)
        except ValidationError as err:
            # Schema problems are converted into structured ERROR responses.
            # This keeps the queue protocol predictable for consumers.
            request_id = self._extract_request_id(body)
            self.logger.warning(
                "Invalid counterfactual request schema for request_id=%s: %s",
                request_id,
                err,
            )
            response = CounterfactualResponse(
                request_id=request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA,
                message="Invalid counterfactual request payload.",
                runtime_ms=int((time.perf_counter() - started) * 1000),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=True,
                    medical_rules_passed=False,
                ),
                timestamp=datetime.now(timezone.utc),
            )
            if not correlation_id:
                correlation_id = request_id
        except Exception as err:
            # Any unexpected runtime failure is also converted into a response
            # payload so the caller is not left waiting forever.
            request_id = self._extract_request_id(body)
            self.logger.exception(
                "Unhandled counterfactual request failure for request_id=%s: %s",
                request_id,
                err,
            )
            response = CounterfactualResponse(
                request_id=request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INTERNAL_ERROR,
                message="Counterfactual service failed while processing request.",
                runtime_ms=int((time.perf_counter() - started) * 1000),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=True,
                    medical_rules_passed=False,
                ),
                timestamp=datetime.now(timezone.utc),
            )
            if not correlation_id:
                correlation_id = request_id

        # Publish before ack so we do not acknowledge successful handling until
        # a response has been prepared and sent.
        self._publish_response(
            response_queue=response_queue, correlation_id=correlation_id, response=response
        )
        self._record_response_metrics(response)
        channel.basic_ack(delivery_tag=method.delivery_tag)

        self.logger.info(
            "cf_request_completed request_id=%s status=%s reason_code=%s "
            "runtime_ms=%s candidate_count=%s engine_version=%s",
            response.request_id,
            response.status.value,
            response.reason_code.value,
            response.runtime_ms,
            1 if response.candidate is not None else 0,
            engine_version,
        )

    def _record_response_metrics(self, response: CounterfactualResponse) -> None:
        self._processed_count += 1
        self._status_counts[response.status.value] += 1
        self._reason_counts[response.reason_code.value] += 1
        self._last_request_id = response.request_id
        self._last_runtime_ms = response.runtime_ms

    def get_health_snapshot(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "rabbitmq_connected": self.connection is not None and self.connection.is_open,
            "request_queue": self.settings.request_queue,
            "response_queue": self.settings.response_queue,
            "processed_count": self._processed_count,
            "status_counts": dict(self._status_counts),
            "reason_counts": dict(self._reason_counts),
            "last_request_id": self._last_request_id,
            "last_runtime_ms": self._last_runtime_ms,
            "engine_version": getattr(self.engine, "engine_version", "unknown"),
        }

    def _cache_response(self, request_id: str, response: CounterfactualResponse) -> None:
        max_size = max(0, self.settings.idempotency_cache_size)
        if max_size == 0 or not request_id:
            return

        self._response_cache[request_id] = response
        self._response_cache.move_to_end(request_id)
        while len(self._response_cache) > max_size:
            self._response_cache.popitem(last=False)

    def _publish_response(
        self,
        response_queue: str,
        correlation_id: str | None,
        response: CounterfactualResponse,
    ) -> None:
        """Serialize and publish one response message to RabbitMQ."""
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel is not initialized.")

        body = json.dumps(response.to_wire()).encode("utf-8")
        max_attempts = max(1, self.settings.rabbitmq_publish_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                self._publish_response_body(
                    response_queue=response_queue,
                    correlation_id=correlation_id,
                    body=body,
                )
                return
            except Exception as err:
                if attempt >= max_attempts:
                    raise
                self.logger.warning(
                    "RabbitMQ response publish attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    err,
                )
                time.sleep(self.settings.rabbitmq_retry_delay_sec)

    def _publish_response_body(
        self,
        *,
        response_queue: str,
        correlation_id: str | None,
        body: bytes,
    ) -> None:
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel is not initialized.")

        self.channel.basic_publish(
            exchange="",
            routing_key=response_queue,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                # Echo the correlation id back so clients using RPC-style
                # messaging can associate the response with the original
                # request.
                correlation_id=correlation_id,
                # delivery_mode=2 marks the message as persistent.
                delivery_mode=2,
                timestamp=int(time.time()),
            ),
        )

    @staticmethod
    def _extract_request_id(body: bytes) -> str:
        """Best-effort extraction of request_id from a raw message body.

        This is only used in failure paths where the main schema validation
        already failed. The helper keeps error responses traceable when possible.
        """
        try:
            payload: Any = json.loads(body.decode("utf-8"))
            request_id = payload.get("request_id")
            if isinstance(request_id, str) and request_id.strip():
                return request_id
        except Exception:
            pass
        return "unknown"
