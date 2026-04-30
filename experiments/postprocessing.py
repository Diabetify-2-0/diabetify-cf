from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureRegistry
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    JSONFeatureValue,
    PredictionInfo,
    ValidationSummary,
)


@dataclass(frozen=True)
class PreparedExperimentRequest:
    registry: FeatureRegistry
    model_columns: list[str]
    instance_features: dict[str, JSONFeatureValue]
    immutable_set: set[str]
    mutable_allowed: list[str]
    query_df: pd.DataFrame
    base_prediction: PredictionInfo
    permitted_range: dict[str, list[float]]


@dataclass(frozen=True)
class ExperimentPostprocessResult:
    status: Status
    reason_code: ReasonCode
    message: str
    runtime_ms: int
    candidates: list[CounterfactualCandidate]
    input_prediction: PredictionInfo | None
    validation: ValidationSummary


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: CounterfactualCandidate
    objective_score: float


class ExperimentPostprocessor:
    """Shared experiment validation, scoring, and ranking for CF generators."""

    def __init__(self, artifacts: ModelArtifacts, max_lof_score: float = 2.5) -> None:
        self.artifacts = artifacts
        self.max_lof_score = max(1.0, float(max_lof_score))

    def prepare(self, request: CounterfactualRequest) -> PreparedExperimentRequest:
        registry = self.artifacts.feature_registry
        model_columns = self.artifacts.feature_columns
        instance_features = registry.canonicalize_feature_map(request.instance.features)
        immutable_input = registry.canonicalize_feature_names(
            request.constraints.immutable_features
        )
        mutable_input = registry.canonicalize_feature_names(request.constraints.mutable_allowed)
        must_not_change = registry.canonicalize_feature_names(request.constraints.must_not_change)

        unknown_features = [name for name in instance_features if name not in model_columns]
        if unknown_features:
            unknown_str = ", ".join(sorted(unknown_features))
            raise ValueError(f"Unknown feature(s) in instance payload: {unknown_str}")

        missing_features = [name for name in model_columns if name not in instance_features]
        if missing_features:
            missing_str = ", ".join(missing_features)
            raise ValueError(f"Missing required feature(s): {missing_str}")

        immutable_set = (
            set(registry.immutable_defaults()).union(immutable_input).union(must_not_change)
        )
        mutable_allowed = self._build_mutable_allowed(
            mutable_input=mutable_input,
            model_columns=model_columns,
            immutable_set=immutable_set,
            registry=registry,
        )
        instance_series = self._to_series(model_columns, instance_features, registry)
        query_df = pd.DataFrame([instance_series], columns=model_columns)
        query_df = self.as_model_input_df(query_df)
        base_prediction = self.predict_info(query_df)
        permitted_range = self._build_permitted_range(
            model_columns=model_columns,
            mutable_allowed=mutable_allowed,
            request=request,
            registry=registry,
            baseline_features=instance_features,
        )

        return PreparedExperimentRequest(
            registry=registry,
            model_columns=model_columns,
            instance_features=instance_features,
            immutable_set=immutable_set,
            mutable_allowed=mutable_allowed,
            query_df=query_df,
            base_prediction=base_prediction,
            permitted_range=permitted_range,
        )

    def no_mutable_result(
        self, request: CounterfactualRequest, started: float
    ) -> ExperimentPostprocessResult:
        return ExperimentPostprocessResult(
            status=Status.INFEASIBLE,
            reason_code=ReasonCode.NO_MUTABLE_FEATURE,
            message="No mutable feature selected after immutable/constraint reconciliation.",
            runtime_ms=self._elapsed_ms(started),
            candidates=[],
            input_prediction=None,
            validation=ValidationSummary(
                immutable_violation=False,
                mutable_compliance=True,
                medical_rules_passed=True,
            ),
        )

    def process(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        raw_candidates: pd.DataFrame,
        started: float,
    ) -> ExperimentPostprocessResult:
        timed_out = self._elapsed_ms(started) > request.generation.timeout_ms
        if raw_candidates.empty:
            reason_code = ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
            message = "No counterfactual candidate generated under current constraints."
            if timed_out:
                reason_code = ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION
                message = "Counterfactual generation exceeded timeout with no feasible solution."
            return ExperimentPostprocessResult(
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                runtime_ms=self._elapsed_ms(started),
                candidates=[],
                input_prediction=prepared.base_prediction,
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )

        evaluated: list[_CandidateEvaluation] = []
        immutable_violation_seen = False
        mutable_violation_seen = False
        medical_violation_seen = False
        directional_violation_seen = False
        target_violation_seen = False

        for _, row in raw_candidates.iterrows():
            candidate_features = self._coerce_candidate_features(
                row=row,
                model_columns=prepared.model_columns,
                registry=prepared.registry,
            )

            immutable_ok = self._immutable_ok(
                candidate_features, prepared.instance_features, prepared.immutable_set
            )
            mutable_ok = self._mutable_ok(
                candidate_features,
                prepared.instance_features,
                set(prepared.mutable_allowed),
            )
            directional_ok = self._directional_ok(
                candidate=candidate_features,
                baseline=prepared.instance_features,
                mutable_allowed=set(prepared.mutable_allowed),
                registry=prepared.registry,
            )
            medical_ok = self._medical_ok(candidate_features, prepared.registry)

            if not immutable_ok:
                immutable_violation_seen = True
            if not mutable_ok:
                mutable_violation_seen = True
            if not directional_ok:
                directional_violation_seen = True
                medical_violation_seen = True
            if not medical_ok:
                medical_violation_seen = True
            if not (immutable_ok and mutable_ok and directional_ok and medical_ok):
                continue

            candidate_df = pd.DataFrame([candidate_features], columns=prepared.model_columns)
            candidate_df = self.as_model_input_df(candidate_df)
            candidate_prediction = self.predict_info(candidate_df)
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
                candidate_features, prepared.instance_features, prepared.registry
            )
            lof_score = self._lof_score(candidate_df)
            if lof_score > self.max_lof_score:
                medical_violation_seen = True
                continue

            candidate = CounterfactualCandidate(
                candidate_id=f"cf_{len(evaluated) + 1}",
                features=candidate_features,
                delta=delta,
                prediction=candidate_prediction,
                metrics=CandidateMetrics(
                    distance_l1=distance_l1,
                    changed_feature_count=len(delta),
                    lof_score=lof_score,
                    constraint_violations=0,
                ),
            )
            evaluated.append(
                _CandidateEvaluation(
                    candidate=candidate,
                    objective_score=self._objective_score(
                        candidate=candidate,
                        preferences=request.preferences.objective_weights,
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
            if (
                medical_violation_seen
                and not immutable_violation_seen
                and not mutable_violation_seen
            ):
                reason_code = ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
                message = "Candidates exist but fail medical plausibility constraints."
            if (
                directional_violation_seen
                and not immutable_violation_seen
                and not mutable_violation_seen
            ):
                reason_code = ReasonCode.MEDICAL_RULE_VIOLATION_ONLY
                message = "Candidates exist but violate directional medical constraints."
            if (
                target_violation_seen
                and not immutable_violation_seen
                and not mutable_violation_seen
                and not medical_violation_seen
            ):
                message = "Candidates generated but none satisfied target class/probability."

            return ExperimentPostprocessResult(
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                runtime_ms=self._elapsed_ms(started),
                candidates=[],
                input_prediction=prepared.base_prediction,
                validation=ValidationSummary(
                    immutable_violation=immutable_violation_seen,
                    mutable_compliance=not mutable_violation_seen,
                    medical_rules_passed=(not medical_violation_seen)
                    and (not target_violation_seen),
                ),
            )

        evaluated.sort(key=lambda item: item.objective_score)
        candidates = [item.candidate for item in evaluated[: request.generation.total_cfs]]
        return ExperimentPostprocessResult(
            status=Status.FEASIBLE,
            reason_code=ReasonCode.OK,
            message=f"Generated {len(candidates)} feasible counterfactual candidate(s).",
            runtime_ms=self._elapsed_ms(started),
            candidates=candidates,
            input_prediction=prepared.base_prediction,
            validation=ValidationSummary(
                immutable_violation=False,
                mutable_compliance=True,
                medical_rules_passed=True,
            ),
        )

    def as_model_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
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

    def predict_info(self, frame: pd.DataFrame) -> PredictionInfo:
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

    def _build_permitted_range(
        self,
        model_columns: list[str],
        mutable_allowed: list[str],
        request: CounterfactualRequest,
        registry: FeatureRegistry,
        baseline_features: dict[str, JSONFeatureValue],
    ) -> dict[str, list[float]]:
        default_range = registry.default_permitted_range(mutable_allowed)
        request_bounds = {}

        for raw_name, bound in request.constraints.feature_bounds.items():
            canonical_name = registry.resolve_name(raw_name)
            if canonical_name not in model_columns or canonical_name not in mutable_allowed:
                continue

            lower = bound.min
            upper = bound.max
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"Conflicting bounds for feature '{canonical_name}': min > max.")

            if canonical_name in default_range:
                current = default_range[canonical_name]
                lower = current[0] if lower is None else lower
                upper = current[1] if upper is None else upper
            elif lower is None or upper is None:
                continue

            request_bounds[canonical_name] = [float(lower), float(upper)]

        merged = dict(default_range)
        merged.update(request_bounds)

        for feature_name in mutable_allowed:
            if feature_name not in merged or feature_name not in baseline_features:
                continue
            feature = registry.get(feature_name)
            if feature is None or feature.preferred_direction == "any":
                continue
            try:
                baseline = float(baseline_features[feature_name])
            except (TypeError, ValueError):
                continue

            lower, upper = merged[feature_name]
            if feature.preferred_direction == "increase":
                lower = max(lower, baseline)
            elif feature.preferred_direction == "decrease":
                upper = min(upper, baseline)
            if lower <= upper:
                merged[feature_name] = [float(lower), float(upper)]

        return merged

    def _build_mutable_allowed(
        self,
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
        return all(
            self._equal(candidate[feature], baseline[feature])
            for feature in immutable_set
            if feature in candidate and feature in baseline
        )

    def _mutable_ok(
        self,
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_set: set[str],
    ) -> bool:
        for feature, value in candidate.items():
            if feature not in baseline:
                continue
            if not self._equal(value, baseline[feature]) and feature not in mutable_set:
                return False
        return True

    def _medical_ok(
        self, candidate: dict[str, JSONFeatureValue], registry: FeatureRegistry
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
        candidate: dict[str, JSONFeatureValue],
        baseline: dict[str, JSONFeatureValue],
        mutable_allowed: set[str],
        registry: FeatureRegistry,
    ) -> bool:
        for feature_name in mutable_allowed:
            if feature_name not in candidate or feature_name not in baseline:
                continue
            feature = registry.get(feature_name)
            if feature is None or feature.preferred_direction == "any":
                continue
            try:
                delta = float(candidate[feature_name]) - float(baseline[feature_name])
            except (TypeError, ValueError):
                return False
            if abs(delta) < 1e-9:
                continue
            if feature.preferred_direction == "increase" and delta < 0:
                return False
            if feature.preferred_direction == "decrease" and delta > 0:
                return False
        return True

    def _target_satisfied(
        self,
        prediction: PredictionInfo,
        target_class: str,
        min_target_probability: float,
    ) -> bool:
        normalized = target_class.strip().lower()
        if normalized in {"low", "low_risk", "non_diabetes", "0"}:
            return (
                prediction.class_name == "low_risk"
                and prediction.probability_low_risk >= min_target_probability
            )
        if normalized in {"high", "high_risk", "diabetes", "1"}:
            high_risk_probability = 1.0 - prediction.probability_low_risk
            return (
                prediction.class_name == "high_risk"
                and high_risk_probability >= min_target_probability
            )
        return prediction.probability_low_risk >= min_target_probability

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
        total = 0.0
        count = 0
        for feature_name, candidate_value in candidate.items():
            if feature_name not in baseline:
                continue
            feature = registry.get(feature_name)
            if feature is None:
                continue
            delta = abs(float(candidate_value) - float(baseline[feature_name]))
            if feature.global_min is not None and feature.global_max is not None:
                denom = max(feature.global_max - feature.global_min, 1e-6)
            else:
                denom = max(abs(float(baseline[feature_name])), 1.0)
            total += delta / denom
            count += 1
        return 0.0 if count == 0 else float(total / count)

    def _lof_score(self, candidate_df: pd.DataFrame) -> float:
        if self.artifacts.lof_model is None:
            return 1.0
        score = float(self.artifacts.lof_model.score_samples(candidate_df)[0])
        return float(max(1e-6, -score))

    def _objective_score(
        self,
        candidate: CounterfactualCandidate,
        preferences: dict[str, float],
        registry: FeatureRegistry,
    ) -> float:
        w_proximity = float(preferences.get("proximity", 0.30))
        w_sparsity = float(preferences.get("sparsity", 0.20))
        w_plausibility = float(preferences.get("plausibility", 0.20))
        w_action_cost = float(preferences.get("action_cost", 0.20))

        proximity_score = candidate.metrics.distance_l1
        sparse_score = float(max(candidate.metrics.changed_feature_count, 1))
        plausibility_score = abs(candidate.metrics.lof_score - 1.0)
        action_cost_score = 0.0
        for feature_name, diff in candidate.delta.items():
            feature = registry.get(feature_name)
            weight = 1.0 if feature is None else feature.cost_weight
            action_cost_score += abs(float(diff)) * weight

        return (
            w_proximity * proximity_score
            + w_sparsity * sparse_score
            + w_plausibility * plausibility_score
            + w_action_cost * action_cost_score
        )

    @staticmethod
    def dice_target_class(target: str, base_prediction: PredictionInfo) -> int | str:
        normalized = target.strip().lower()
        if normalized in {"low", "low_risk", "non_diabetes", "0"}:
            return 0
        if normalized in {"high", "high_risk", "diabetes", "1"}:
            return 1
        return 1 if base_prediction.class_name == "low_risk" else 0

    @staticmethod
    def normalize_dice_method(method: str) -> str:
        mapping = {
            "dice_genetic": "genetic",
            "genetic": "genetic",
            "dice_random": "random",
            "random": "random",
            "dice_kdtree": "kdtree",
            "kdtree": "kdtree",
        }
        return mapping.get(method.lower(), "genetic")

    @staticmethod
    def _equal(a: object, b: object) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        return a == b

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)
