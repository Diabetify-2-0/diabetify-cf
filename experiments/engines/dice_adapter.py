from __future__ import annotations

from diabetify_cf.config import Settings
from diabetify_cf.engine import DiceCounterfactualEngine
from diabetify_cf.schemas import CounterfactualRequest
from experiments.engines.base import EngineRunResult, ExperimentEngine


class DiceExperimentAdapter(ExperimentEngine):
    name = "dice"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.engine = DiceCounterfactualEngine(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
            max_lof_score=self.settings.max_lof_score,
            planner=None,
        )

    def generate(self, request: CounterfactualRequest) -> EngineRunResult:
        try:
            response = self.engine.generate(request)
            return EngineRunResult(
                engine_name=self.name,
                request_id=response.request_id,
                status=response.status.value,
                reason_code=response.reason_code.value,
                message=response.message,
                runtime_ms=response.runtime_ms,
                candidate_count=len(response.candidates),
                candidates=[candidate.to_wire() for candidate in response.candidates],
            )
        except Exception as err:
            return EngineRunResult(
                engine_name=self.name,
                request_id=request.request_id,
                status="ERROR",
                reason_code="EXPERIMENT_ADAPTER_ERROR",
                message=str(err),
                runtime_ms=0,
                candidate_count=0,
                raw_error=repr(err),
            )
