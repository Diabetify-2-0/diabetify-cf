from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from diabetify_cf.schemas import CounterfactualRequest


@dataclass(frozen=True)
class EngineRunResult:
    engine_name: str
    request_id: str
    status: str
    reason_code: str
    message: str
    runtime_ms: int
    candidate_count: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_error: str | None = None

    def to_case_record(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "request_id": self.request_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "message": self.message,
            "runtime_ms": self.runtime_ms,
            "candidate_count": self.candidate_count,
            "raw_error": self.raw_error,
        }


class ExperimentEngine(ABC):
    name: str

    @abstractmethod
    def generate(self, request: CounterfactualRequest) -> EngineRunResult:
        raise NotImplementedError
