from __future__ import annotations

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
class NearestNeighborOptions:
    candidate_pool_size: int = 256
    max_neighbors: int = 64
    max_changed_features: int = 3
    min_reference_low_risk_probability: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> NearestNeighborOptions:
        raw = (config or {}).get("engine_options") or {}
        if not isinstance(raw, dict):
            raise ValueError("engine_options must be an object.")

        candidate_pool_size = int(raw.get("candidate_pool_size", cls.candidate_pool_size))
        if candidate_pool_size < 1:
            raise ValueError("NN engine option 'candidate_pool_size' must be >= 1.")

        max_neighbors = int(raw.get("max_neighbors", cls.max_neighbors))
        if max_neighbors < 1:
            raise ValueError("NN engine option 'max_neighbors' must be >= 1.")

        max_changed_features = int(raw.get("max_changed_features", cls.max_changed_features))
        if max_changed_features < 1:
            raise ValueError("NN engine option 'max_changed_features' must be >= 1.")

        raw_min_probability = raw.get("min_reference_low_risk_probability")
        min_reference_low_risk_probability = None
        if raw_min_probability is not None:
            min_reference_low_risk_probability = float(raw_min_probability)
            if not 0.0 <= min_reference_low_risk_probability <= 1.0:
                raise ValueError(
                    "NN engine option 'min_reference_low_risk_probability' must be within [0, 1]."
                )

        return cls(
            candidate_pool_size=candidate_pool_size,
            max_neighbors=max_neighbors,
            max_changed_features=max_changed_features,
            min_reference_low_risk_probability=min_reference_low_risk_probability,
        )


class NearestNeighborCandidateGenerator:
    """Reference-data nearest-neighbor baseline for actionable counterfactuals."""

    engine_version = "nearest_neighbor_projection_experiment_v1"

    def __init__(
        self,
        model_path: str,
        columns_path: str,
        reference_data_path: str,
        feature_registry_path: str,
        options: NearestNeighborOptions | None = None,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        self.options = options or NearestNeighborOptions()
        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
        except Exception as exc:
            self.artifacts = None
            self.initialization_error = str(exc)

    def generate_raw(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> pd.DataFrame:
        assert self.artifacts is not None

        baseline = prepared.instance_features
        mutable_allowed = prepared.mutable_allowed
        if not mutable_allowed:
            return pd.DataFrame()

        reference_frame = self.artifacts.reference_data[self.artifacts.feature_columns].copy()
        reference_frame = postprocessor.as_model_input_df(reference_frame)
        low_risk_probabilities = self.artifacts.model.predict_proba(reference_frame)[:, 0]

        min_probability = self.options.min_reference_low_risk_probability
        if min_probability is None:
            min_probability = float(request.target.min_target_probability)
        eligible_mask = low_risk_probabilities >= min_probability
        eligible = reference_frame.loc[eligible_mask].copy()
        eligible_probabilities = low_risk_probabilities[eligible_mask]

        if eligible.empty:
            top_indices = np.argsort(low_risk_probabilities)[::-1][
                : self.options.candidate_pool_size
            ]
            eligible = reference_frame.iloc[top_indices].copy()
            eligible_probabilities = low_risk_probabilities[top_indices]
        elif len(eligible) > self.options.candidate_pool_size:
            ranked = np.argsort(eligible_probabilities)[::-1][: self.options.candidate_pool_size]
            eligible = eligible.iloc[ranked].copy()
            eligible_probabilities = eligible_probabilities[ranked]

        ranked_neighbors = self._rank_neighbors(
            eligible=eligible,
            eligible_probabilities=eligible_probabilities,
            baseline=baseline,
            mutable_allowed=mutable_allowed,
            prepared=prepared,
        )
        projected = self._project_neighbors(
            ranked_neighbors=ranked_neighbors[: self.options.max_neighbors],
            baseline=baseline,
            mutable_allowed=mutable_allowed,
            prepared=prepared,
        )
        if not projected:
            return pd.DataFrame()

        raw = pd.DataFrame(projected).drop_duplicates()
        ordered = raw[self.artifacts.feature_columns]
        return postprocessor.as_model_input_df(ordered)

    def _rank_neighbors(
        self,
        *,
        eligible: pd.DataFrame,
        eligible_probabilities: np.ndarray,
        baseline: dict[str, Any],
        mutable_allowed: list[str],
        prepared: PreparedExperimentRequest,
    ) -> list[pd.Series]:
        ranked: list[tuple[tuple[float, float, int], pd.Series]] = []
        for row, low_risk_probability in zip(
            eligible.itertuples(index=False, name=None),
            eligible_probabilities,
            strict=True,
        ):
            series = pd.Series(row, index=eligible.columns)
            weighted_distance = 0.0
            changed_count = 0
            for feature_name in mutable_allowed:
                delta = abs(float(series[feature_name]) - float(baseline[feature_name]))
                if delta < 1e-9:
                    continue
                changed_count += 1
                weighted_distance += self._weighted_normalized_delta(
                    feature_name=feature_name,
                    delta=delta,
                    prepared=prepared,
                )
            score = (-float(low_risk_probability), weighted_distance, changed_count)
            ranked.append((score, series))
        ranked.sort(key=lambda item: item[0])
        return [series for _, series in ranked]

    def _project_neighbors(
        self,
        *,
        ranked_neighbors: list[pd.Series],
        baseline: dict[str, Any],
        mutable_allowed: list[str],
        prepared: PreparedExperimentRequest,
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for neighbor in ranked_neighbors:
            changed_features = self._rank_changed_features(
                neighbor=neighbor,
                baseline=baseline,
                mutable_allowed=mutable_allowed,
                prepared=prepared,
            )
            if not changed_features:
                continue

            full_projection = dict(baseline)
            for feature_name in changed_features:
                full_projection[feature_name] = neighbor[feature_name]
            self._append_projection(
                projection=full_projection,
                projected=projected,
                seen=seen,
            )

            sparse_limit = min(self.options.max_changed_features, len(changed_features))
            for changed_count in range(1, sparse_limit + 1):
                sparse_projection = dict(baseline)
                for feature_name in changed_features[:changed_count]:
                    sparse_projection[feature_name] = neighbor[feature_name]
                self._append_projection(
                    projection=sparse_projection,
                    projected=projected,
                    seen=seen,
                )
        return projected

    def _rank_changed_features(
        self,
        *,
        neighbor: pd.Series,
        baseline: dict[str, Any],
        mutable_allowed: list[str],
        prepared: PreparedExperimentRequest,
    ) -> list[str]:
        changed: list[tuple[float, str]] = []
        for feature_name in mutable_allowed:
            delta = abs(float(neighbor[feature_name]) - float(baseline[feature_name]))
            if delta < 1e-9:
                continue
            changed.append(
                (
                    self._weighted_normalized_delta(
                        feature_name=feature_name,
                        delta=delta,
                        prepared=prepared,
                    ),
                    feature_name,
                )
            )
        changed.sort(key=lambda item: (item[0], item[1]))
        return [feature_name for _, feature_name in changed]

    def _weighted_normalized_delta(
        self,
        *,
        feature_name: str,
        delta: float,
        prepared: PreparedExperimentRequest,
    ) -> float:
        feature = prepared.registry.get(feature_name)
        cost_weight = 1.0 if feature is None else float(feature.cost_weight)
        span = self._feature_span(feature_name=feature_name, feature=feature, prepared=prepared)
        return (delta / span) * cost_weight

    def _feature_span(
        self,
        *,
        feature_name: str,
        feature: FeatureDefinition | None,
        prepared: PreparedExperimentRequest,
    ) -> float:
        if feature_name in prepared.permitted_range:
            lower, upper = prepared.permitted_range[feature_name]
            return max(float(upper) - float(lower), 1e-6)
        if (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            return max(float(feature.global_max) - float(feature.global_min), 1e-6)
        assert self.artifacts is not None
        series = pd.to_numeric(
            self.artifacts.reference_data[feature_name],
            errors="coerce",
        ).dropna()
        if series.empty:
            return 1.0
        return max(float(series.max() - series.min()), 1e-6)

    def _append_projection(
        self,
        *,
        projection: dict[str, Any],
        projected: list[dict[str, Any]],
        seen: set[tuple[Any, ...]],
    ) -> None:
        key = self._state_key(projection)
        if key in seen:
            return
        seen.add(key)
        projected.append(projection)

    def _state_key(self, state: dict[str, Any]) -> tuple[Any, ...]:
        assert self.artifacts is not None
        key_parts: list[Any] = []
        for column in self.artifacts.feature_columns:
            value = state[column]
            if isinstance(value, (int, float)):
                key_parts.append(round(float(value), 6))
            else:
                key_parts.append(value)
        return tuple(key_parts)


class NearestNeighborExperimentAdapter(ExperimentEngine):
    name = "nn"

    def __init__(
        self,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.engine = NearestNeighborCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
            options=NearestNeighborOptions.from_config(config),
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
            message = "Experiment NN generator is not fully configured yet."
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
