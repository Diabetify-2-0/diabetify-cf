from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from diabetify_cf.config import Settings
from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest, ValidationSummary
from experiments.engines.base import EngineRunResult, ExperimentEngine
from experiments.postprocessing import (
    ExperimentPostprocessor,
    ExperimentPostprocessResult,
    PreparedExperimentRequest,
)


class OceanCandidateGenerator:
    """OCEAN CP candidate generator for tree-ensemble counterfactual comparisons."""

    engine_version = "ocean_cp_experiment_v1"

    def __init__(
        self,
        model_path: str,
        columns_path: str,
        reference_data_path: str,
        feature_registry_path: str,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
            self._check_ocean_available()
        except Exception as exc:
            self.artifacts = None
            self.initialization_error = str(exc)

    @staticmethod
    def _check_ocean_available() -> None:
        import ocean  # noqa: F401

    def generate_raw(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> pd.DataFrame:
        assert self.artifacts is not None

        from ocean import ConstraintProgrammingExplainer

        mapper = self._build_mapper()
        explainer = ConstraintProgrammingExplainer(self.artifacts.model, mapper=mapper)
        x = prepared.query_df[self.artifacts.feature_columns].iloc[0].to_numpy(dtype=float)
        target_class = self._target_class(request.target.target_class)
        max_time = max(1, math.ceil(request.generation.timeout_ms / 1000))

        explanation = explainer.explain(
            x,
            y=target_class,
            norm=1,
            max_time=max_time,
            random_seed=request.generation.random_seed,
            verbose=False,
        )
        if explanation is None:
            return pd.DataFrame()

        raw = pd.DataFrame(
            [np.asarray(explanation.x, dtype=float)],
            columns=self.artifacts.feature_columns,
        )
        return postprocessor.as_model_input_df(raw)

    def _build_mapper(self) -> Any:
        assert self.artifacts is not None

        from ocean.abc import Mapper
        from ocean.feature import Feature

        mapping = {
            column: self._to_ocean_feature(column, Feature)
            for column in self.artifacts.feature_columns
        }
        columns = pd.MultiIndex.from_tuples(
            [(column, "") for column in self.artifacts.feature_columns]
        )
        return Mapper(mapping, columns=columns)

    def _to_ocean_feature(self, column: str, feature_cls: Any) -> Any:
        assert self.artifacts is not None
        feature = self.artifacts.feature_registry.get(column)
        if feature is not None and feature.is_binary:
            return feature_cls(feature_cls.Type.BINARY)
        if feature is not None and feature.feature_type == "ordinal":
            return feature_cls(
                feature_cls.Type.DISCRETE,
                levels=self._discrete_levels(column, feature),
            )
        return feature_cls(
            feature_cls.Type.CONTINUOUS,
            levels=self._continuous_bounds(column, feature),
        )

    def _continuous_bounds(
        self,
        column: str,
        feature: FeatureDefinition | None,
    ) -> list[float]:
        assert self.artifacts is not None
        if (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            return [float(feature.global_min), float(feature.global_max)]
        series = self.artifacts.reference_data[column]
        return [float(series.min()), float(series.max())]

    def _discrete_levels(self, column: str, feature: FeatureDefinition) -> list[float]:
        assert self.artifacts is not None
        if (
            feature.global_min is not None
            and feature.global_max is not None
            and float(feature.global_max - feature.global_min) <= 100
        ):
            start = int(feature.global_min)
            stop = int(feature.global_max)
            return [float(value) for value in range(start, stop + 1)]
        values = self.artifacts.reference_data[column].dropna().unique()
        return sorted(float(value) for value in values)

    @staticmethod
    def _target_class(target_class: str) -> int:
        normalized = target_class.strip().lower()
        if normalized in {"low_risk", "low", "0"}:
            return 0
        if normalized in {"high_risk", "high", "1"}:
            return 1
        raise ValueError(f"Unsupported OCEAN target class: {target_class}")


class OceanExperimentAdapter(ExperimentEngine):
    name = "ocean"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.engine = OceanCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
        )

    @property
    def artifacts(self) -> ModelArtifacts | None:
        return self.engine.artifacts

    @property
    def initialization_error(self) -> str | None:
        return self.engine.initialization_error

    def generate(self, request: CounterfactualRequest) -> EngineRunResult:
        started = perf_counter()
        if self.artifacts is None:
            message = "Experiment OCEAN generator is not fully configured yet."
            if self.initialization_error:
                message += f" Initialization error: {self.initialization_error}"
            return self._to_run_result(
                request=request,
                result=ExperimentPostprocessResult(
                    status=Status.ERROR,
                    reason_code=ReasonCode.ENGINE_NOT_READY,
                    message=message,
                    runtime_ms=self._elapsed_ms(started),
                    candidates=[],
                    input_prediction=None,
                    validation=ValidationSummary(
                        immutable_violation=False,
                        mutable_compliance=True,
                        medical_rules_passed=True,
                    ),
                ),
            )

        postprocessor = ExperimentPostprocessor(
            artifacts=self.artifacts,
            max_lof_score=self.settings.max_lof_score,
        )
        try:
            prepared = postprocessor.prepare(request)
            if not prepared.mutable_allowed:
                return self._to_run_result(
                    request=request,
                    result=postprocessor.no_mutable_result(request, started),
                )
            raw_candidates = self.engine.generate_raw(
                request=request,
                prepared=prepared,
                postprocessor=postprocessor,
            )
            result = postprocessor.process(
                request=request,
                prepared=prepared,
                raw_candidates=raw_candidates,
                started=started,
            )
            return self._to_run_result(request=request, result=result)
        except ValueError as err:
            return EngineRunResult(
                engine_name=self.name,
                request_id=request.request_id,
                status=Status.ERROR.value,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA.value,
                message=str(err),
                runtime_ms=self._elapsed_ms(started),
                candidate_count=0,
                candidates=[],
                raw_error=repr(err),
            )
        except Exception as err:
            return EngineRunResult(
                engine_name=self.name,
                request_id=request.request_id,
                status=Status.ERROR.value,
                reason_code=ReasonCode.INTERNAL_ERROR.value,
                message=f"Unhandled experiment generator error: {err}",
                runtime_ms=self._elapsed_ms(started),
                candidate_count=0,
                candidates=[],
                raw_error=repr(err),
            )

    def _to_run_result(
        self,
        *,
        request: CounterfactualRequest,
        result: ExperimentPostprocessResult,
    ) -> EngineRunResult:
        return EngineRunResult(
            engine_name=self.name,
            request_id=request.request_id,
            status=result.status.value,
            reason_code=result.reason_code.value,
            message=result.message,
            runtime_ms=result.runtime_ms,
            candidate_count=len(result.candidates),
            candidates=[candidate.to_wire() for candidate in result.candidates],
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)
