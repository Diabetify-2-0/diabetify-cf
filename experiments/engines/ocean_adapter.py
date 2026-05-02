from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class OceanSolverOptions:
    norm: int = 1
    attempt_count: int = 1
    seed_step: int = 997
    max_time_per_attempt_seconds: int | None = None
    num_workers: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> OceanSolverOptions:
        raw = (config or {}).get("engine_options") or {}
        if not isinstance(raw, dict):
            raise ValueError("engine_options must be an object.")

        norm = int(raw.get("norm", cls.norm))
        if norm < 0:
            raise ValueError("OCEAN engine option 'norm' must be >= 0.")

        attempt_count = int(raw.get("attempt_count", cls.attempt_count))
        if attempt_count < 1:
            raise ValueError("OCEAN engine option 'attempt_count' must be >= 1.")

        seed_step = int(raw.get("seed_step", cls.seed_step))
        if seed_step < 1:
            raise ValueError("OCEAN engine option 'seed_step' must be >= 1.")

        max_time_raw = raw.get("max_time_per_attempt_seconds")
        max_time_per_attempt_seconds = (
            None if max_time_raw is None else int(max_time_raw)
        )
        if (
            max_time_per_attempt_seconds is not None
            and max_time_per_attempt_seconds < 1
        ):
            raise ValueError(
                "OCEAN engine option 'max_time_per_attempt_seconds' must be >= 1."
            )

        num_workers_raw = raw.get("num_workers")
        num_workers = None if num_workers_raw is None else int(num_workers_raw)
        if num_workers is not None and num_workers < 1:
            raise ValueError("OCEAN engine option 'num_workers' must be >= 1.")

        return cls(
            norm=norm,
            attempt_count=attempt_count,
            seed_step=seed_step,
            max_time_per_attempt_seconds=max_time_per_attempt_seconds,
            num_workers=num_workers,
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
        solver_options: OceanSolverOptions | None = None,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        self.solver_options = solver_options or OceanSolverOptions()
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

        mapper = self._build_mapper(prepared)
        explainer = ConstraintProgrammingExplainer(self.artifacts.model, mapper=mapper)
        x = prepared.query_df[self.artifacts.feature_columns].iloc[0].to_numpy(dtype=float)
        target_class = self._target_class(request.target.target_class)
        max_time = self._max_time_per_attempt(request)

        raw_candidates: list[pd.DataFrame] = []
        try:
            for attempt_index in range(self.solver_options.attempt_count):
                explanation = explainer.explain(
                    x,
                    y=target_class,
                    norm=self.solver_options.norm,
                    max_time=max_time,
                    random_seed=self._attempt_seed(request, attempt_index),
                    verbose=False,
                    **self._worker_kwargs(),
                )
                if explanation is None:
                    continue

                raw_candidates.append(
                    pd.DataFrame(
                        [np.asarray(explanation.x, dtype=float)],
                        columns=self.artifacts.feature_columns,
                    )
                )
        except IndexError:
            return pd.DataFrame()

        if not raw_candidates:
            return pd.DataFrame()

        raw = pd.concat(raw_candidates, ignore_index=True).drop_duplicates()
        return postprocessor.as_model_input_df(raw)

    def _max_time_per_attempt(self, request: CounterfactualRequest) -> int:
        total_budget = max(1, math.ceil(request.generation.timeout_ms / 1000))
        configured = self.solver_options.max_time_per_attempt_seconds
        if configured is not None:
            return max(1, min(configured, total_budget))
        return max(1, math.ceil(total_budget / self.solver_options.attempt_count))

    def _attempt_seed(self, request: CounterfactualRequest, attempt_index: int) -> int:
        return request.generation.random_seed + (
            attempt_index * self.solver_options.seed_step
        )

    def _worker_kwargs(self) -> dict[str, int]:
        if self.solver_options.num_workers is None:
            return {}
        return {"num_workers": self.solver_options.num_workers}

    def _build_mapper(self, prepared: PreparedExperimentRequest) -> Any:
        assert self.artifacts is not None

        from ocean.abc import Mapper
        from ocean.feature import Feature

        mapping = {
            column: self._to_ocean_feature(column, Feature, prepared)
            for column in self.artifacts.feature_columns
        }
        columns = pd.MultiIndex.from_tuples(
            [(column, "") for column in self.artifacts.feature_columns]
        )
        return Mapper(mapping, columns=columns)

    def _to_ocean_feature(
        self,
        column: str,
        feature_cls: Any,
        prepared: PreparedExperimentRequest,
    ) -> Any:
        assert self.artifacts is not None
        feature = self.artifacts.feature_registry.get(column)
        lower, upper = self._request_bounds(column, feature, prepared)
        if lower == upper:
            return feature_cls(feature_cls.Type.DISCRETE, levels=[lower])

        if feature is not None and feature.is_binary:
            return feature_cls(feature_cls.Type.BINARY)
        if feature is not None and feature.feature_type == "ordinal":
            return feature_cls(
                feature_cls.Type.DISCRETE,
                levels=self._discrete_levels(column, feature, lower=lower, upper=upper),
            )
        return feature_cls(
            feature_cls.Type.CONTINUOUS,
            levels=[lower, upper],
        )

    def _request_bounds(
        self,
        column: str,
        feature: FeatureDefinition | None,
        prepared: PreparedExperimentRequest,
    ) -> tuple[float, float]:
        baseline = float(prepared.instance_features[column])
        if column not in prepared.mutable_allowed:
            return baseline, baseline

        if column in prepared.permitted_range:
            lower, upper = prepared.permitted_range[column]
        elif (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            lower, upper = float(feature.global_min), float(feature.global_max)
        else:
            series = self.artifacts.reference_data[column]
            lower, upper = float(series.min()), float(series.max())

        lower = float(lower)
        upper = float(upper)
        if lower > upper:
            return baseline, baseline
        return lower, upper

    def _discrete_levels(
        self,
        column: str,
        feature: FeatureDefinition,
        *,
        lower: float,
        upper: float,
    ) -> list[float]:
        assert self.artifacts is not None
        if (
            feature.global_min is not None
            and feature.global_max is not None
            and float(feature.global_max - feature.global_min) <= 100
        ):
            start = math.ceil(max(float(feature.global_min), lower))
            stop = math.floor(min(float(feature.global_max), upper))
            if start > stop:
                return [float(round(lower))]
            return [float(value) for value in range(start, stop + 1)]
        values = self.artifacts.reference_data[column].dropna().unique()
        levels = sorted(float(value) for value in values if lower <= float(value) <= upper)
        return levels or [float(lower)]

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

    def __init__(
        self,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.engine = OceanCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
            solver_options=OceanSolverOptions.from_config(config),
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
