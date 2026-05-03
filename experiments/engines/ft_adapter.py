from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

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
class FeatureTweakOptions:
    max_changed_features: int = 2
    beam_width: int = 24
    max_candidates_to_evaluate: int = 300
    max_thresholds_per_feature: int = 16
    threshold_epsilon: float = 1e-4
    search_patience: int = 2

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> FeatureTweakOptions:
        raw = (config or {}).get("engine_options") or {}
        if not isinstance(raw, dict):
            raise ValueError("engine_options must be an object.")

        max_changed_features = int(raw.get("max_changed_features", cls.max_changed_features))
        if max_changed_features < 1:
            raise ValueError("FT engine option 'max_changed_features' must be >= 1.")

        beam_width = int(raw.get("beam_width", cls.beam_width))
        if beam_width < 1:
            raise ValueError("FT engine option 'beam_width' must be >= 1.")

        max_candidates_to_evaluate = int(
            raw.get("max_candidates_to_evaluate", cls.max_candidates_to_evaluate)
        )
        if max_candidates_to_evaluate < 1:
            raise ValueError("FT engine option 'max_candidates_to_evaluate' must be >= 1.")

        max_thresholds_per_feature = int(
            raw.get("max_thresholds_per_feature", cls.max_thresholds_per_feature)
        )
        if max_thresholds_per_feature < 1:
            raise ValueError("FT engine option 'max_thresholds_per_feature' must be >= 1.")

        threshold_epsilon = float(raw.get("threshold_epsilon", cls.threshold_epsilon))
        if threshold_epsilon <= 0.0:
            raise ValueError("FT engine option 'threshold_epsilon' must be > 0.")

        search_patience = int(raw.get("search_patience", cls.search_patience))
        if search_patience < 1:
            raise ValueError("FT engine option 'search_patience' must be >= 1.")

        return cls(
            max_changed_features=max_changed_features,
            beam_width=beam_width,
            max_candidates_to_evaluate=max_candidates_to_evaluate,
            max_thresholds_per_feature=max_thresholds_per_feature,
            threshold_epsilon=threshold_epsilon,
            search_patience=search_patience,
        )


class FeatureTweakCandidateGenerator:
    """Threshold-search baseline inspired by FeatureTweak for tree ensembles."""

    engine_version = "feature_tweak_style_experiment_v1"

    def __init__(
        self,
        model_path: str,
        columns_path: str,
        reference_data_path: str,
        feature_registry_path: str,
        options: FeatureTweakOptions | None = None,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        self.options = options or FeatureTweakOptions()
        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
            self._feature_thresholds = self._extract_feature_thresholds()
        except Exception as exc:
            self.artifacts = None
            self.initialization_error = str(exc)
            self._feature_thresholds = {}

    def generate_raw(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> pd.DataFrame:
        assert self.artifacts is not None

        base_state = {
            column: prepared.instance_features[column] for column in self.artifacts.feature_columns
        }
        frontier = [base_state]
        visited = {self._state_key(base_state)}
        collected: list[dict[str, Any]] = []
        evaluated_count = 0
        stale_depths = 0
        target_low_risk = self._target_low_risk(request.target.target_class)
        started = perf_counter()

        for depth in range(1, self.options.max_changed_features + 1):
            if self._timed_out(started, request.generation.timeout_ms):
                break

            expanded = self._expand_frontier(
                frontier=frontier,
                prepared=prepared,
                visited=visited,
            )
            if not expanded:
                break

            remaining_budget = self.options.max_candidates_to_evaluate - evaluated_count
            if remaining_budget <= 0:
                break
            if len(expanded) > remaining_budget:
                expanded = expanded[:remaining_budget]

            evaluated_count += len(expanded)
            frame = self._states_to_frame(expanded, postprocessor)
            probabilities = self.artifacts.model.predict_proba(frame)

            ranked_states = []
            new_feasible_count = 0
            for state, probability in zip(expanded, probabilities, strict=True):
                low_risk_probability = float(probability[0])
                score = self._state_score(
                    state=state,
                    baseline=prepared.instance_features,
                    prepared=prepared,
                    low_risk_probability=low_risk_probability,
                    target_low_risk=target_low_risk,
                )
                ranked_states.append((score, state, low_risk_probability))
                if low_risk_probability >= request.target.min_target_probability:
                    collected.append(state)
                    new_feasible_count += 1

            ranked_states.sort(key=lambda item: item[0])
            frontier = [state for _, state, _ in ranked_states[: self.options.beam_width]]

            if new_feasible_count == 0:
                stale_depths += 1
                if stale_depths >= self.options.search_patience and depth >= 1:
                    break
            else:
                stale_depths = 0

            if len(collected) >= max(request.generation.total_cfs * 4, self.options.beam_width):
                break

        if not collected:
            return pd.DataFrame()

        raw = pd.DataFrame(collected).drop_duplicates()
        ordered = raw[self.artifacts.feature_columns]
        return postprocessor.as_model_input_df(ordered)

    def _extract_feature_thresholds(self) -> dict[str, list[float]]:
        assert self.artifacts is not None
        booster = self._resolve_booster(self.artifacts.model)
        trees = booster.trees_to_dataframe()
        if "Feature" not in trees.columns or "Split" not in trees.columns:
            raise ValueError("XGBoost tree dump is missing Feature/Split columns.")

        thresholds: dict[str, set[float]] = {
            column: set() for column in self.artifacts.feature_columns
        }
        for row in trees.itertuples(index=False):
            feature_name = getattr(row, "Feature", None)
            if not isinstance(feature_name, str) or feature_name == "Leaf":
                continue
            if feature_name not in thresholds:
                continue
            split_value = getattr(row, "Split", None)
            if split_value is None or not math.isfinite(float(split_value)):
                continue
            thresholds[feature_name].add(float(split_value))

        return {
            feature_name: sorted(values)
            for feature_name, values in thresholds.items()
            if values
        }

    @staticmethod
    def _resolve_booster(model: Any) -> Any:
        if hasattr(model, "get_booster"):
            return model.get_booster()
        if hasattr(model, "trees_to_dataframe"):
            return model
        raise ValueError("FT-style engine requires an XGBoost model with tree access.")

    def _expand_frontier(
        self,
        *,
        frontier: list[dict[str, Any]],
        prepared: PreparedExperimentRequest,
        visited: set[tuple[Any, ...]],
    ) -> list[dict[str, Any]]:
        assert self.artifacts is not None
        expanded: list[dict[str, Any]] = []
        for state in frontier:
            changed_features = {
                feature_name
                for feature_name, value in state.items()
                if feature_name in prepared.mutable_allowed
                and abs(float(value) - float(prepared.instance_features[feature_name])) >= 1e-9
            }
            for feature_name in prepared.mutable_allowed:
                if feature_name in changed_features:
                    continue
                feature = self.artifacts.feature_registry.get(feature_name)
                lower, upper = self._bounds_for_feature(feature_name, feature, prepared)
                baseline_value = float(prepared.instance_features[feature_name])
                for candidate_value in self._candidate_values_for_feature(
                    feature_name=feature_name,
                    feature=feature,
                    baseline_value=baseline_value,
                    lower=lower,
                    upper=upper,
                ):
                    if abs(candidate_value - float(state[feature_name])) < 1e-9:
                        continue
                    child = dict(state)
                    child[feature_name] = self._coerce_feature_value(
                        candidate_value=candidate_value,
                        feature_name=feature_name,
                        feature=feature,
                    )
                    key = self._state_key(child)
                    if key in visited:
                        continue
                    visited.add(key)
                    expanded.append(child)
        return expanded

    def _candidate_values_for_feature(
        self,
        *,
        feature_name: str,
        feature: FeatureDefinition | None,
        baseline_value: float,
        lower: float,
        upper: float,
    ) -> list[float]:
        values: list[float] = []
        if feature is not None and feature.is_binary:
            values = [lower, upper]
        elif feature is not None and feature.feature_type == "ordinal":
            values = self._ordinal_candidate_values(
                feature_name=feature_name,
                baseline_value=baseline_value,
                lower=lower,
                upper=upper,
            )
        else:
            values = self._continuous_candidate_values(
                feature_name=feature_name,
                baseline_value=baseline_value,
                lower=lower,
                upper=upper,
            )

        deduped: list[float] = []
        seen: set[float] = set()
        for value in values:
            clipped = min(max(value, lower), upper)
            rounded = round(float(clipped), 6)
            if abs(rounded - baseline_value) < 1e-9 or rounded in seen:
                continue
            deduped.append(rounded)
            seen.add(rounded)
        return deduped[: self.options.max_thresholds_per_feature]

    def _continuous_candidate_values(
        self,
        *,
        feature_name: str,
        baseline_value: float,
        lower: float,
        upper: float,
    ) -> list[float]:
        values = [lower, upper]
        for threshold in self._feature_thresholds.get(feature_name, []):
            if threshold < lower or threshold > upper:
                continue
            if threshold >= baseline_value:
                values.append(threshold + self.options.threshold_epsilon)
            if threshold <= baseline_value:
                values.append(threshold - self.options.threshold_epsilon)
        values.sort(key=lambda value: (abs(value - baseline_value), value))
        return values

    def _ordinal_candidate_values(
        self,
        *,
        feature_name: str,
        baseline_value: float,
        lower: float,
        upper: float,
    ) -> list[float]:
        candidates: set[int] = set()
        for threshold in self._feature_thresholds.get(feature_name, []):
            floor_value = math.floor(threshold)
            ceil_value = math.ceil(threshold)
            if lower <= floor_value <= upper:
                candidates.add(floor_value)
            if lower <= ceil_value <= upper:
                candidates.add(ceil_value)
        lower_int = int(math.ceil(lower))
        upper_int = int(math.floor(upper))
        candidates.update({lower_int, upper_int})
        values = sorted(candidates, key=lambda value: (abs(value - baseline_value), value))
        return [float(value) for value in values]

    def _bounds_for_feature(
        self,
        feature_name: str,
        feature: FeatureDefinition | None,
        prepared: PreparedExperimentRequest,
    ) -> tuple[float, float]:
        baseline = float(prepared.instance_features[feature_name])
        if feature_name not in prepared.mutable_allowed:
            return baseline, baseline
        if feature_name in prepared.permitted_range:
            lower, upper = prepared.permitted_range[feature_name]
            return float(lower), float(upper)
        if (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            return float(feature.global_min), float(feature.global_max)
        assert self.artifacts is not None
        series = self.artifacts.reference_data[feature_name]
        return float(series.min()), float(series.max())

    def _state_score(
        self,
        *,
        state: dict[str, Any],
        baseline: dict[str, Any],
        prepared: PreparedExperimentRequest,
        low_risk_probability: float,
        target_low_risk: bool,
    ) -> tuple[float, float, int]:
        target_probability = low_risk_probability if target_low_risk else 1.0 - low_risk_probability
        changed_count = 0
        weighted_distance = 0.0
        for feature_name in prepared.mutable_allowed:
            delta = abs(float(state[feature_name]) - float(baseline[feature_name]))
            if delta < 1e-9:
                continue
            changed_count += 1
            feature = prepared.registry.get(feature_name)
            cost_weight = 1.0 if feature is None else float(feature.cost_weight)
            weighted_distance += delta * cost_weight
        return (-target_probability, weighted_distance, changed_count)

    def _states_to_frame(
        self, states: list[dict[str, Any]], postprocessor: ExperimentPostprocessor
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        frame = pd.DataFrame(states, columns=self.artifacts.feature_columns)
        return postprocessor.as_model_input_df(frame)

    def _coerce_feature_value(
        self,
        *,
        candidate_value: float,
        feature_name: str,
        feature: FeatureDefinition | None,
    ) -> Any:
        assert self.artifacts is not None
        if feature is None:
            return candidate_value
        return self.artifacts.feature_registry.coerce_value(feature_name, candidate_value)

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

    @staticmethod
    def _target_low_risk(target_class: str) -> bool:
        normalized = target_class.strip().lower()
        return normalized not in {"high", "high_risk", "diabetes", "1"}

    @staticmethod
    def _timed_out(started: float, timeout_ms: int) -> bool:
        return ((perf_counter() - started) * 1000) > timeout_ms


class FeatureTweakExperimentAdapter(ExperimentEngine):
    name = "ft"

    def __init__(
        self,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.engine = FeatureTweakCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
            options=FeatureTweakOptions.from_config(config),
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
            message = "Experiment FT-style generator is not fully configured yet."
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
