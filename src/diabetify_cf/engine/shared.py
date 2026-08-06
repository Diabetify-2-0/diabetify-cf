from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.engine.feature_registry import FeatureRegistry
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    JSONFeatureValue,
    PlannerFeatureChange,
    PlannerInput,
    PredictionInfo,
    ValidationSummary,
)


@dataclass(frozen=True)
class PreparedRequest:
    registry: FeatureRegistry
    model_columns: list[str]
    instance_features: dict[str, JSONFeatureValue]
    immutable_set: set[str]
    mutable_allowed: list[str]
    query_df: pd.DataFrame
    base_prediction: PredictionInfo


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CounterfactualCandidate
    objective_score: float


class ArtifactBackedCounterfactualEngine(CounterfactualEngine, ABC):
    """Shared request preparation and candidate validation workflow."""

    engine_version = "artifact_backed_engine_v1"

    def __init__(
        self,
        *,
        model_path: str = "",
        columns_path: str = "",
        reference_data_path: str = "",
        feature_registry_path: str = "",
    ) -> None:
        self.logger = logging.getLogger("diabetify_cf.engine")
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        self._reference_lof_scores_cache: np.ndarray | None = None

        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
        except Exception as exc:
            self.initialization_error = str(exc)

    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        started = perf_counter()

        if len(request.constraints.mutable_allowed) == 0:
            return self._response(
                request=request,
                started=started,
                status=Status.INFEASIBLE,
                reason_code=ReasonCode.NO_MUTABLE_FEATURE,
                message="No mutable feature selected by user.",
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=False,
                    medical_rules_passed=True,
                ),
            )

        if self.artifacts is None:
            if self.initialization_error:
                self.logger.error(
                    "%s initialization failed: %s",
                    self.__class__.__name__,
                    self.initialization_error,
                )
            return self._response(
                request=request,
                started=started,
                status=Status.ERROR,
                reason_code=ReasonCode.ENGINE_NOT_READY,
                message="Counterfactual engine is not ready.",
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=False,
                    medical_rules_passed=True,
                ),
            )

        try:
            prepared = self._prepare_request(request)
            request_context = self._build_request_context_planner_input(
                request=request,
                prepared=prepared,
            )
            if self._target_satisfied(
                prediction=prepared.base_prediction,
                target_class=request.target.target_class,
                min_target_probability=request.target.min_target_probability,
            ):
                return self._response(
                    request=request,
                    started=started,
                    status=Status.FEASIBLE,
                    reason_code=ReasonCode.TARGET_ALREADY_SATISFIED,
                    message=(
                        "Input already satisfies target class/probability; "
                        "no counterfactual changes are required."
                    ),
                    validation=ValidationSummary(
                        immutable_violation=False,
                        mutable_violation=False,
                        medical_rules_passed=True,
                    ),
                    input_prediction=prepared.base_prediction,
                    planner_input=request_context,
                )
            if not prepared.mutable_allowed:
                return self._response(
                    request=request,
                    started=started,
                    status=Status.INFEASIBLE,
                    reason_code=ReasonCode.NO_MUTABLE_FEATURE,
                    message=(
                        "No mutable feature selected after immutable/constraint " "reconciliation."
                    ),
                    validation=ValidationSummary(
                        immutable_violation=False,
                        mutable_violation=False,
                        medical_rules_passed=True,
                    ),
                    input_prediction=prepared.base_prediction,
                    planner_input=request_context,
                )

            raw_candidates = self._generate_raw_candidates_with_timeout(
                request=request,
                prepared=prepared,
                started=started,
            )
            return self._process_candidates(
                request=request,
                prepared=prepared,
                raw_candidates=raw_candidates,
                started=started,
            )
        except ValueError as err:
            self.logger.warning(
                "Invalid counterfactual request for request_id=%s: %s",
                request.request_id,
                err,
            )
            return self._response(
                request=request,
                started=started,
                status=Status.ERROR,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA,
                message="Invalid counterfactual request payload.",
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=True,
                    medical_rules_passed=False,
                ),
            )
        except TimeoutError:
            planner_input = PlannerInput()
            input_prediction = None
            if "prepared" in locals():
                input_prediction = prepared.base_prediction
                planner_input = self._build_request_context_planner_input(
                    request=request,
                    prepared=prepared,
                )
            return self._response(
                request=request,
                started=started,
                status=Status.INFEASIBLE,
                reason_code=ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION,
                message="Counterfactual generation exceeded request timeout.",
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=False,
                    medical_rules_passed=False,
                ),
                input_prediction=input_prediction,
                planner_input=planner_input,
            )
        except Exception:
            self.logger.exception(
                "Unhandled counterfactual engine error for request_id=%s",
                request.request_id,
            )
            return self._response(
                request=request,
                started=started,
                status=Status.ERROR,
                reason_code=ReasonCode.INTERNAL_ERROR,
                message="Counterfactual engine failed while processing request.",
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=True,
                    medical_rules_passed=False,
                ),
            )

    @abstractmethod
    def _generate_raw_candidates(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _generate_raw_candidates_with_timeout(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
        started: float,
    ) -> pd.DataFrame:
        remaining_ms = request.generation.timeout_ms - self._elapsed_ms(started)
        if remaining_ms <= 0:
            raise TimeoutError

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cf-engine")
        future = executor.submit(
            self._generate_raw_candidates,
            request=request,
            prepared=prepared,
        )
        try:
            result = future.result(timeout=remaining_ms / 1000.0)
        except FutureTimeoutError as err:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError from err
        except Exception:
            executor.shutdown(wait=True, cancel_futures=False)
            raise

        executor.shutdown(wait=True, cancel_futures=False)
        return result

    def _prepare_request(self, request: CounterfactualRequest) -> PreparedRequest:
        assert self.artifacts is not None
        registry = self.artifacts.feature_registry
        model_columns = self.artifacts.feature_columns

        instance_features = registry.canonicalize_feature_map(request.instance.features)
        mutable_input = registry.canonicalize_feature_names(request.constraints.mutable_allowed)

        unknown_features = [name for name in instance_features if name not in model_columns]
        if unknown_features:
            unknown_str = ", ".join(sorted(unknown_features))
            raise ValueError(f"Unknown feature(s) in instance payload: {unknown_str}")

        missing_features = [name for name in model_columns if name not in instance_features]
        if missing_features:
            missing_str = ", ".join(missing_features)
            raise ValueError(f"Missing required feature(s): {missing_str}")

        self._validate_instance_feature_values(
            instance_features=instance_features,
            registry=registry,
        )

        immutable_set = set(registry.immutable_defaults())
        mutable_allowed = self._build_mutable_allowed(
            mutable_input=mutable_input,
            model_columns=model_columns,
            immutable_set=immutable_set,
            registry=registry,
        )
        query_df = pd.DataFrame(
            [self._to_series(model_columns, instance_features, registry)],
            columns=model_columns,
        )
        query_df = self._as_model_input_df(query_df)
        base_prediction = self._predict_info(query_df)
        return PreparedRequest(
            registry=registry,
            model_columns=model_columns,
            instance_features=instance_features,
            immutable_set=immutable_set,
            mutable_allowed=mutable_allowed,
            query_df=query_df,
            base_prediction=base_prediction,
        )

    def _process_candidates(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
        raw_candidates: pd.DataFrame,
        started: float,
    ) -> CounterfactualResponse:
        timed_out = self._elapsed_ms(started) > request.generation.timeout_ms
        request_context = self._build_request_context_planner_input(
            request=request,
            prepared=prepared,
        )
        if raw_candidates.empty:
            reason_code = ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
            message = "No counterfactual candidate generated under current constraints."
            if timed_out:
                reason_code = ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION
                message = "Counterfactual generation exceeded timeout with no feasible solution."
            return self._response(
                request=request,
                started=started,
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=False,
                    medical_rules_passed=True,
                ),
                input_prediction=prepared.base_prediction,
                planner_input=request_context,
            )

        evaluated: list[CandidateEvaluation] = []
        medical_violation_seen = False
        directional_violation_seen = False
        transition_violation_seen = False
        target_violation_seen = False

        for _, row in raw_candidates.iterrows():
            candidate_features = self._coerce_candidate_features(
                row=row,
                model_columns=prepared.model_columns,
                registry=prepared.registry,
            )

            directional_ok = self._directional_ok(
                candidate=candidate_features,
                baseline=prepared.instance_features,
                mutable_allowed=set(prepared.mutable_allowed),
                registry=prepared.registry,
            )
            transition_ok = self._transition_ok(
                candidate=candidate_features,
                baseline=prepared.instance_features,
                mutable_allowed=set(prepared.mutable_allowed),
                registry=prepared.registry,
            )
            medical_ok = self._medical_ok(candidate_features, prepared.registry)

            if not directional_ok:
                directional_violation_seen = True
                medical_violation_seen = True
            if not transition_ok:
                transition_violation_seen = True
                medical_violation_seen = True
            if not medical_ok:
                medical_violation_seen = True

            if not (directional_ok and transition_ok and medical_ok):
                continue

            candidate_df = pd.DataFrame([candidate_features], columns=prepared.model_columns)
            candidate_df = self._as_model_input_df(candidate_df)
            candidate_prediction = self._predict_info(candidate_df)
            if not self._target_satisfied(
                prediction=candidate_prediction,
                target_class=request.target.target_class,
                min_target_probability=request.target.min_target_probability,
            ):
                target_violation_seen = True
                continue

            delta = self._build_delta(
                candidate_features,
                prepared.instance_features,
                set(prepared.mutable_allowed),
            )
            distance_l1 = self._normalized_l1(
                candidate_features,
                prepared.instance_features,
                prepared.registry,
            )
            lof_score = self._lof_score(candidate_df)

            candidate = CounterfactualCandidate(
                candidate_id=f"cf_{len(evaluated) + 1}",
                features=candidate_features,
                delta=delta,
                prediction=candidate_prediction,
                metrics=CandidateMetrics(
                    distance_l1=distance_l1,
                    changed_feature_count=len(delta),
                    lof_score=lof_score,
                ),
            )
            evaluated.append(
                CandidateEvaluation(
                    candidate=candidate,
                    objective_score=self._objective_score(
                        candidate=candidate,
                        preferences={},
                        mutable_allowed=prepared.mutable_allowed,
                        registry=prepared.registry,
                    ),
                )
            )

        if not evaluated:
            reason_code = ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
            message = "No valid counterfactual candidate after constraint validation."
            if timed_out:
                reason_code = ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION
                message = (
                    "Counterfactual generation exceeded timeout before finding valid candidates."
                )
            if medical_violation_seen:
                reason_code = ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
                message = "Candidates exist but fail medical plausibility constraints."
            if transition_violation_seen:
                reason_code = ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
                message = "Candidates exist but violate allowed feature transition constraints."
            if directional_violation_seen:
                reason_code = ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
                message = "Candidates exist but violate directional medical constraints."
            if target_violation_seen and not medical_violation_seen:
                message = "Candidates generated but none satisfied target class/probability."

            return self._response(
                request=request,
                started=started,
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_violation=False,
                    medical_rules_passed=(not medical_violation_seen)
                    and (not target_violation_seen),
                ),
                input_prediction=prepared.base_prediction,
                planner_input=request_context,
            )

        evaluated.sort(key=lambda item: item.objective_score)
        top_candidate = evaluated[0].candidate
        planner_input = self._build_planner_input(
            request=request,
            prepared=prepared,
            candidate=top_candidate,
        )

        return self._response(
            request=request,
            started=started,
            status=Status.FEASIBLE,
            reason_code=ReasonCode.OK,
            message="Generated a feasible counterfactual candidate.",
            validation=ValidationSummary(
                immutable_violation=False,
                mutable_violation=False,
                medical_rules_passed=True,
            ),
            input_prediction=prepared.base_prediction,
            candidate=top_candidate,
            planner_input=planner_input,
        )

    def _response(
        self,
        *,
        request: CounterfactualRequest,
        started: float,
        status: Status,
        reason_code: ReasonCode,
        message: str,
        validation: ValidationSummary,
        input_prediction: PredictionInfo | None = None,
        candidate: CounterfactualCandidate | None = None,
        planner_input: PlannerInput | None = None,
    ) -> CounterfactualResponse:
        return CounterfactualResponse(
            request_id=request.request_id,
            status=status,
            reason_code=reason_code,
            message=message,
            runtime_ms=self._elapsed_ms(started),
            input_prediction=input_prediction,
            candidate=candidate,
            validation=validation,
            planner_input=planner_input or PlannerInput(),
        )

    def _as_model_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.artifacts is not None
        typed = frame.copy()
        registry = self.artifacts.feature_registry

        for column in self.artifacts.feature_columns:
            if column not in typed.columns:
                continue
            numeric = pd.to_numeric(typed[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    f"Feature '{column}' contains non-numeric value(s) after coercion."
                )

            feature = registry.get(column)
            if feature is not None and (feature.is_binary or feature.feature_type == "ordinal"):
                typed[column] = numeric.round().astype("int64")
            else:
                typed[column] = numeric.astype("float64")

        return typed

    def _validate_instance_feature_values(
        self,
        *,
        instance_features: dict[str, JSONFeatureValue],
        registry: FeatureRegistry,
    ) -> None:
        for feature_name, value in instance_features.items():
            feature = registry.get(feature_name)
            if feature is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as err:
                raise ValueError(f"Feature '{feature_name}' must be numeric.") from err

            if np.isnan(numeric_value):
                raise ValueError(f"Feature '{feature_name}' cannot be NaN.")
            if feature.global_min is not None and numeric_value < feature.global_min:
                raise ValueError(f"Feature '{feature_name}' is below minimum {feature.global_min}.")
            if feature.global_max is not None and numeric_value > feature.global_max:
                raise ValueError(f"Feature '{feature_name}' is above maximum {feature.global_max}.")
            if feature.is_binary and numeric_value not in {0.0, 1.0}:
                raise ValueError(f"Feature '{feature_name}' must be binary 0/1.")

    def _predict_info(self, frame: pd.DataFrame) -> PredictionInfo:
        assert self.artifacts is not None

        if hasattr(self.artifacts.model, "predict_proba"):
            probabilities = self.artifacts.model.predict_proba(frame)[0]
            if len(probabilities) >= 2:
                proba_high = float(probabilities[1])
            else:
                proba_high = float(probabilities[0])
        else:
            proba_high = float(self.artifacts.model.predict(frame)[0])

        proba_high = float(np.clip(proba_high, 0.0, 1.0))
        proba_low = float(np.clip(1.0 - proba_high, 0.0, 1.0))
        class_name = "low_risk" if proba_low >= 0.5 else "high_risk"
        return PredictionInfo.model_validate(
            {"class": class_name, "probability_low_risk": proba_low}
        )

    def _target_satisfied(
        self,
        prediction: PredictionInfo,
        target_class: str,
        min_target_probability: float,
    ) -> bool:
        return prediction.probability_low_risk >= min_target_probability

    def _build_mutable_allowed(
        self,
        *,
        mutable_input: list[str],
        model_columns: list[str],
        immutable_set: set[str],
        registry: FeatureRegistry,
    ) -> list[str]:
        allowed: list[str] = []
        seen: set[str] = set()
        for name in mutable_input:
            if name in seen or name not in model_columns or name in immutable_set:
                continue

            feature = registry.get(name)
            if feature is not None and not feature.actionable:
                continue

            allowed.append(name)
            seen.add(name)
        return allowed

    def _to_series(
        self,
        ordered_columns: list[str],
        features: dict[str, JSONFeatureValue],
        registry: FeatureRegistry,
    ) -> dict[str, object]:
        series: dict[str, object] = {}
        for column in ordered_columns:
            value = features[column]
            if isinstance(value, bool):
                value = int(value)
            series[column] = registry.coerce_value(column, value)
        return series

    def _coerce_candidate_features(
        self,
        *,
        row: pd.Series,
        model_columns: list[str],
        registry: FeatureRegistry,
    ) -> dict[str, JSONFeatureValue]:
        result: dict[str, JSONFeatureValue] = {}
        for column in model_columns:
            value = row[column]
            if pd.isna(value):
                value = 0.0
            coerced = registry.coerce_value(column, value)
            if not isinstance(coerced, (int, float, bool, str)):
                coerced = float(str(coerced))
            result[column] = coerced
        return result

    def _immutable_ok(
        self,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        immutable_set: set[str],
    ) -> bool:
        for feature in immutable_set:
            if feature not in candidate or feature not in baseline:
                continue
            if not self._equal(candidate[feature], baseline[feature]):
                return False
        return True

    def _mutable_ok(
        self,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_set: set[str],
    ) -> bool:
        for feature, value in candidate.items():
            if feature not in baseline:
                continue
            changed = not self._equal(value, baseline[feature])
            if changed and feature not in mutable_set:
                return False
        return True

    def _medical_ok(
        self,
        candidate: dict[str, JSONFeatureValue],
        registry: FeatureRegistry,
    ) -> bool:
        for feature_name, value in candidate.items():
            feature = registry.get(feature_name)
            if feature is None:
                continue

            try:
                val = float(value)
            except (TypeError, ValueError):
                return False
            if np.isnan(val):
                return False
            if feature.global_min is not None and val < feature.global_min:
                return False
            if feature.global_max is not None and val > feature.global_max:
                return False
            if feature.is_binary and val not in {0.0, 1.0}:
                return False
        return True

    def _directional_ok(
        self,
        *,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_allowed: set[str],
        registry: FeatureRegistry,
    ) -> bool:
        for feature_name in mutable_allowed:
            if feature_name not in candidate or feature_name not in baseline:
                continue

            feature = registry.get(feature_name)
            if feature is None:
                continue

            direction = feature.preferred_direction
            if direction == "any":
                continue

            try:
                delta = float(candidate[feature_name]) - float(baseline[feature_name])
            except (TypeError, ValueError):
                return False

            if abs(delta) < 1e-9:
                continue
            if direction == "increase" and delta < 0:
                return False
            if direction == "decrease" and delta > 0:
                return False
        return True

    def _transition_ok(
        self,
        *,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_allowed: set[str],
        registry: FeatureRegistry,
    ) -> bool:
        for feature_name in mutable_allowed:
            if feature_name not in candidate or feature_name not in baseline:
                continue

            feature = registry.get(feature_name)
            if feature is None or not feature.allowed_transitions:
                continue

            try:
                baseline_value = int(float(baseline[feature_name]))
                candidate_value = int(float(candidate[feature_name]))
            except (TypeError, ValueError):
                return False

            allowed_targets = feature.allowed_transitions.get(baseline_value)
            if allowed_targets is None:
                continue
            if candidate_value not in allowed_targets:
                return False

        return True

    def _build_delta(
        self,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_allowed: set[str],
    ) -> dict[str, float]:
        delta: dict[str, float] = {}
        for feature in mutable_allowed:
            if feature not in candidate or feature not in baseline:
                continue
            diff = float(candidate[feature]) - float(baseline[feature])
            if abs(diff) >= 1e-9:
                delta[feature] = round(diff, 6)
        return delta

    def _normalized_l1(
        self,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        registry: FeatureRegistry,
    ) -> float:
        squared_total = 0.0
        for feature_name, candidate_value in candidate.items():
            if feature_name not in baseline:
                continue
            feature = registry.get(feature_name)
            if feature is None:
                continue

            if feature.is_binary:
                component = 0.0 if self._equal(candidate_value, baseline[feature_name]) else 1.0
            else:
                delta = abs(float(candidate_value) - float(baseline[feature_name]))
                if feature.global_min is not None and feature.global_max is not None:
                    denom = max(feature.global_max - feature.global_min, 1e-6)
                else:
                    denom = max(
                        abs(float(candidate_value)),
                        abs(float(baseline[feature_name])),
                        1.0,
                    )
                component = delta / denom

            squared_total += component**2

        return float(np.sqrt(squared_total))

    def _lof_score(self, candidate_df: pd.DataFrame) -> float:
        assert self.artifacts is not None
        if self.artifacts.lof_model is None:
            return 1.0

        score = float(self.artifacts.lof_model.score_samples(candidate_df.to_numpy())[0])
        return float(max(1e-6, -score))

    def _objective_score(
        self,
        *,
        candidate: CounterfactualCandidate,
        preferences: dict[str, float],
        mutable_allowed: list[str],
        registry: FeatureRegistry,
    ) -> float:
        w_proximity = float(preferences.get("proximity", 0.75))
        w_plausibility = float(preferences.get("plausibility", 0.25))

        mutable_count = max(len(mutable_allowed), 1)
        proximity_denominator = max(float(np.sqrt(mutable_count)), 1e-6)
        proximity_score = min(
            max(float(candidate.metrics.distance_l1) / proximity_denominator, 0.0),
            1.0,
        )
        plausibility_score = self._normalized_lof_score(candidate.metrics.lof_score)

        return w_proximity * proximity_score + w_plausibility * plausibility_score

    def _normalized_lof_score(self, lof_score: float) -> float:
        reference_scores = self._reference_lof_scores()
        if reference_scores.size == 0:
            return min(max(float(lof_score) / max(float(lof_score), 1.0), 0.0), 1.0)
        rank = np.searchsorted(reference_scores, float(lof_score), side="right")
        return float(rank / reference_scores.size)

    def _reference_lof_scores(self) -> np.ndarray:
        if self._reference_lof_scores_cache is not None:
            return self._reference_lof_scores_cache
        lof_model = getattr(self.artifacts, "lof_model", None)
        if self.artifacts is None or lof_model is None:
            self._reference_lof_scores_cache = np.asarray([], dtype=float)
            return self._reference_lof_scores_cache

        training_scores = self._training_lof_scores(lof_model)
        if training_scores.size > 0:
            self._reference_lof_scores_cache = training_scores
            return self._reference_lof_scores_cache

        reference_df = self.artifacts.reference_data[self.artifacts.feature_columns].copy()
        if reference_df.empty:
            self._reference_lof_scores_cache = np.asarray([], dtype=float)
            return self._reference_lof_scores_cache

        reference_df = self._as_model_input_df(reference_df)
        raw_scores = lof_model.score_samples(reference_df.to_numpy())
        lof_scores = np.maximum(1e-6, -np.asarray(raw_scores, dtype=float))
        self._reference_lof_scores_cache = np.sort(lof_scores)
        return self._reference_lof_scores_cache

    def _training_lof_scores(self, lof_model: object) -> np.ndarray:
        named_steps = getattr(lof_model, "named_steps", {})
        steps = list(named_steps.values()) if hasattr(named_steps, "values") else [lof_model]
        for step in reversed(steps):
            negative_factors = getattr(step, "negative_outlier_factor_", None)
            if negative_factors is None:
                continue
            lof_scores = np.maximum(1e-6, -np.asarray(negative_factors, dtype=float))
            return np.sort(lof_scores)
        return np.asarray([], dtype=float)

    def _build_planner_input(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
        candidate: CounterfactualCandidate,
    ) -> PlannerInput:
        changed_features: list[PlannerFeatureChange] = []
        for feature_name, delta in sorted(
            candidate.delta.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        ):
            changed_features.append(
                PlannerFeatureChange(
                    feature_name=feature_name,
                    baseline_value=prepared.instance_features[feature_name],
                    candidate_value=candidate.features[feature_name],
                    delta=delta,
                    direction="decrease" if float(delta) < 0 else "increase",
                )
            )

        return PlannerInput(
            recommended_candidate_id=candidate.candidate_id,
            target_deltas=candidate.delta,
            input_prediction=prepared.base_prediction,
            candidate_prediction=candidate.prediction,
            candidate_metrics=candidate.metrics,
            changed_features=changed_features,
            mutable_allowed=list(prepared.mutable_allowed),
        )

    def _build_request_context_planner_input(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
    ) -> PlannerInput:
        return PlannerInput(
            input_prediction=prepared.base_prediction,
            mutable_allowed=list(prepared.mutable_allowed),
        )

    @staticmethod
    def _equal(a: object, b: object) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        return a == b

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)
