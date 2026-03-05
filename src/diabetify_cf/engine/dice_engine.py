from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    PlannerInput,
    PredictionInfo,
    ValidationSummary,
)


@dataclass
class CandidateEvaluation:
    candidate: CounterfactualCandidate
    objective_score: float


class DiceCounterfactualEngine(CounterfactualEngine):
    """
    Phase-3 engine:
    - Supports real DiCE generation when artifacts are configured.
    - Keeps stub mode for controlled plumbing tests.
    """

    def __init__(
        self,
        allow_stub_feasible: bool = False,
        model_path: str = "",
        columns_path: str = "",
        reference_data_path: str = "",
        feature_registry_path: str = "",
        max_lof_score: float = 2.5,
        planner: PrescriptivePlanner | None = None,
    ) -> None:
        self.allow_stub_feasible = allow_stub_feasible
        self.max_lof_score = max(1.0, float(max_lof_score))
        self.engine_version = "dice_engine_v3"
        self.planner = planner

        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None

        if not self.allow_stub_feasible:
            try:
                self.artifacts = load_artifacts(
                    model_path=model_path,
                    columns_path=columns_path,
                    reference_data_path=reference_data_path,
                    feature_registry_path=feature_registry_path,
                )
            except Exception as exc:  # noqa: BLE001
                self.initialization_error = str(exc)

    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        started = perf_counter()

        mutable_allowed = request.constraints.mutable_allowed
        if len(mutable_allowed) == 0:
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.INFEASIBLE,
                reason_code=ReasonCode.NO_MUTABLE_FEATURE,
                message="No mutable feature selected by user.",
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )

        if self.allow_stub_feasible:
            try:
                candidate, deltas = self._build_stub_candidate(request)
            except ValueError as err:
                return CounterfactualResponse(
                    request_id=request.request_id,
                    status=Status.INFEASIBLE,
                    reason_code=ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,
                    message=str(err),
                    model_version=request.model_version,
                    cf_engine_version=self.engine_version,
                    runtime_ms=self._elapsed_ms(started),
                    validation=ValidationSummary(
                        immutable_violation=False,
                        mutable_compliance=False,
                        medical_rules_passed=True,
                    ),
                )

            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.FEASIBLE,
                reason_code=ReasonCode.OK,
                message="Stub feasible counterfactual generated for integration testing.",
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                input_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.2),
                candidates=[candidate],
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
                planner_input=PlannerInput(
                    recommended_candidate_id=candidate.candidate_id,
                    target_deltas=deltas,
                ),
                prescriptive_plan=self._build_prescriptive_plan(
                    request=request,
                    candidate=candidate,
                ),
            )

        if self.artifacts is None:
            message = "DICE engine is not fully configured yet."
            if self.initialization_error:
                message += f" Initialization error: {self.initialization_error}"
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.ENGINE_NOT_READY,
                message=message,
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )

        try:
            return self._generate_real(request=request, started=started)
        except ValueError as err:
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA,
                message=str(err),
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=False,
                    medical_rules_passed=False,
                ),
            )
        except Exception as err:  # noqa: BLE001
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.ERROR,
                reason_code=ReasonCode.INTERNAL_ERROR,
                message=f"Unhandled counterfactual engine error: {err}",
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=False,
                    medical_rules_passed=False,
                ),
            )

    def _generate_real(
        self, request: CounterfactualRequest, started: float
    ) -> CounterfactualResponse:
        assert self.artifacts is not None
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
        if not mutable_allowed:
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.INFEASIBLE,
                reason_code=ReasonCode.NO_MUTABLE_FEATURE,
                message="No mutable feature selected after immutable/constraint reconciliation.",
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )

        instance_series = self._to_series(model_columns, instance_features, registry)
        query_df = pd.DataFrame([instance_series], columns=model_columns)
        base_prediction = self._predict_info(query_df)

        dice_data = self._build_dice_data()
        method = self._normalize_dice_method(request.generation.method)
        desired_class = self._target_to_dice_class(
            target=request.target.target_class,
            base_prediction=base_prediction,
        )
        permitted_range = self._build_permitted_range(
            model_columns=model_columns,
            mutable_allowed=mutable_allowed,
            request=request,
            registry=registry,
            baseline_features=instance_features,
        )

        raw_candidates = self._run_dice(
            dice_data=dice_data,
            method=method,
            query_df=query_df,
            total_cfs=request.generation.total_cfs,
            desired_class=desired_class,
            features_to_vary=mutable_allowed,
            permitted_range=permitted_range,
            random_seed=request.generation.random_seed,
        )
        timed_out = self._elapsed_ms(started) > request.generation.timeout_ms
        if raw_candidates.empty:
            reason_code = ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS
            message = "No counterfactual candidate generated under current constraints."
            if timed_out:
                reason_code = ReasonCode.TIMEOUT_NO_FEASIBLE_SOLUTION
                message = "Counterfactual generation exceeded timeout with no feasible solution."
            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                input_prediction=base_prediction,
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )

        evaluated: List[CandidateEvaluation] = []
        immutable_violation_seen = False
        mutable_violation_seen = False
        medical_violation_seen = False
        directional_violation_seen = False
        target_violation_seen = False

        for _, row in raw_candidates.iterrows():
            candidate_features = self._coerce_candidate_features(
                row=row,
                model_columns=model_columns,
                registry=registry,
            )

            immutable_ok = self._immutable_ok(candidate_features, instance_features, immutable_set)
            mutable_ok = self._mutable_ok(
                candidate_features, instance_features, set(mutable_allowed)
            )
            directional_ok = self._directional_ok(
                candidate=candidate_features,
                baseline=instance_features,
                mutable_allowed=set(mutable_allowed),
                registry=registry,
            )
            medical_ok = self._medical_ok(candidate_features, registry)

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

            candidate_df = pd.DataFrame([candidate_features], columns=model_columns)
            candidate_df = self._as_dice_input_df(candidate_df)
            candidate_prediction = self._predict_info(candidate_df)
            if not self._target_satisfied(
                prediction=candidate_prediction,
                target_class=request.target.target_class,
                min_target_probability=request.target.min_target_probability,
            ):
                target_violation_seen = True
                continue

            delta = self._build_delta(candidate_features, instance_features, set(mutable_allowed))
            distance_l1 = self._normalized_l1(candidate_features, instance_features, registry)
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
            score = self._objective_score(
                candidate=candidate,
                preferences=request.preferences.objective_weights,
                registry=registry,
            )
            evaluated.append(CandidateEvaluation(candidate=candidate, objective_score=score))

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

            return CounterfactualResponse(
                request_id=request.request_id,
                status=Status.INFEASIBLE,
                reason_code=reason_code,
                message=message,
                model_version=request.model_version,
                cf_engine_version=self.engine_version,
                runtime_ms=self._elapsed_ms(started),
                input_prediction=base_prediction,
                validation=ValidationSummary(
                    immutable_violation=immutable_violation_seen,
                    mutable_compliance=not mutable_violation_seen,
                    medical_rules_passed=(not medical_violation_seen)
                    and (not target_violation_seen),
                ),
            )

        evaluated.sort(key=lambda item: item.objective_score)
        candidates = [item.candidate for item in evaluated[: request.generation.total_cfs]]
        top_candidate = candidates[0]

        return CounterfactualResponse(
            request_id=request.request_id,
            status=Status.FEASIBLE,
            reason_code=ReasonCode.OK,
            message=f"Generated {len(candidates)} feasible counterfactual candidate(s).",
            model_version=request.model_version,
            cf_engine_version=self.engine_version,
            runtime_ms=self._elapsed_ms(started),
            input_prediction=base_prediction,
            candidates=candidates,
            validation=ValidationSummary(
                immutable_violation=False,
                mutable_compliance=True,
                medical_rules_passed=True,
            ),
            planner_input=PlannerInput(
                recommended_candidate_id=top_candidate.candidate_id,
                target_deltas=top_candidate.delta,
            ),
            prescriptive_plan=self._build_prescriptive_plan(
                request=request,
                candidate=top_candidate,
            ),
        )

    def _run_dice(
        self,
        dice_data: object,
        method: str,
        query_df: pd.DataFrame,
        total_cfs: int,
        desired_class: int | str,
        features_to_vary: List[str],
        permitted_range: Dict[str, List[float]],
        random_seed: int,
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        import dice_ml

        dice_model = dice_ml.Model(
            model=self.artifacts.model, backend="sklearn", model_type="classifier"
        )
        dice = dice_ml.Dice(dice_data, dice_model, method=method)
        query_for_dice = self._as_dice_input_df(query_df)

        kwargs = {
            "query_instances": query_for_dice,
            "total_CFs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
            "verbose": False,
            "random_seed": random_seed,
        }
        if permitted_range:
            kwargs["permitted_range"] = permitted_range

        try:
            result = dice.generate_counterfactuals(**kwargs)
        except TypeError:
            kwargs.pop("random_seed", None)
            result = dice.generate_counterfactuals(**kwargs)
        cf_examples = result.cf_examples_list[0]
        if cf_examples.final_cfs_df is None:
            return pd.DataFrame()

        raw = cf_examples.final_cfs_df.copy()
        model_columns = self.artifacts.feature_columns
        present = [column for column in model_columns if column in raw.columns]
        if not present:
            return pd.DataFrame()
        return raw[present]

    def _build_dice_data(self) -> object:
        assert self.artifacts is not None
        import dice_ml

        outcome_name = "_cf_outcome"
        reference = self.artifacts.reference_data.copy()
        reference = self._as_dice_input_df(reference)
        reference[outcome_name] = self.artifacts.model.predict(
            reference[self.artifacts.feature_columns]
        )
        reference = self._as_dice_input_df(reference)
        # Treat every model feature as numeric continuous to avoid object/categorical
        # dataframe dtypes that are incompatible with XGBoost in sklearn backend mode.
        continuous = list(self.artifacts.feature_columns)

        return dice_ml.Data(
            dataframe=reference,
            continuous_features=continuous,
            outcome_name=outcome_name,
        )

    def _as_dice_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
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

    def _build_permitted_range(
        self,
        model_columns: List[str],
        mutable_allowed: List[str],
        request: CounterfactualRequest,
        registry: object,
        baseline_features: Dict[str, object],
    ) -> Dict[str, List[float]]:
        default_range = registry.default_permitted_range(mutable_allowed)
        request_bounds = {}

        for raw_name, bound in request.constraints.feature_bounds.items():
            canonical_name = registry.resolve_name(raw_name)
            if canonical_name not in model_columns:
                continue
            if canonical_name not in mutable_allowed:
                continue

            lower = bound.min
            upper = bound.max
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"Conflicting bounds for feature '{canonical_name}': min > max.")

            if canonical_name in default_range:
                current = default_range[canonical_name]
                if lower is None:
                    lower = current[0]
                if upper is None:
                    upper = current[1]
            elif lower is None or upper is None:
                continue

            if lower is not None and upper is not None:
                request_bounds[canonical_name] = [float(lower), float(upper)]

        merged = dict(default_range)
        merged.update(request_bounds)

        for feature_name in mutable_allowed:
            if feature_name not in merged:
                continue
            if feature_name not in baseline_features:
                continue

            feature = registry.get(feature_name)
            if feature is None:
                continue

            direction = getattr(feature, "preferred_direction", "any")
            if direction == "any":
                continue

            try:
                baseline = float(baseline_features[feature_name])
            except (TypeError, ValueError):
                continue

            lower, upper = merged[feature_name]
            if direction == "increase":
                lower = max(lower, baseline)
            elif direction == "decrease":
                upper = min(upper, baseline)

            if lower > upper:
                continue
            merged[feature_name] = [float(lower), float(upper)]

        return merged

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
        return PredictionInfo(class_name=class_name, probability_low_risk=proba_low)

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

    def _target_to_dice_class(self, target: str, base_prediction: PredictionInfo) -> int | str:
        normalized = target.strip().lower()
        if normalized in {"low", "low_risk", "non_diabetes", "0"}:
            return 0
        if normalized in {"high", "high_risk", "diabetes", "1"}:
            return 1
        if base_prediction.class_name == "low_risk":
            return 1
        return 0

    def _normalize_dice_method(self, method: str) -> str:
        mapping = {
            "dice_genetic": "genetic",
            "genetic": "genetic",
            "dice_random": "random",
            "random": "random",
            "dice_kdtree": "kdtree",
            "kdtree": "kdtree",
        }
        return mapping.get(method.lower(), "genetic")

    def _build_mutable_allowed(
        self,
        mutable_input: List[str],
        model_columns: List[str],
        immutable_set: set[str],
        registry: object,
    ) -> List[str]:
        allowed: List[str] = []
        seen: set[str] = set()
        for name in mutable_input:
            if name in seen:
                continue
            if name not in model_columns:
                continue
            if name in immutable_set:
                continue

            feature = registry.get(name)
            if feature is not None and not feature.actionable:
                continue

            allowed.append(name)
            seen.add(name)
        return allowed

    def _to_series(
        self, ordered_columns: List[str], features: Dict[str, object], registry: object
    ) -> Dict[str, object]:
        series: Dict[str, object] = {}
        for column in ordered_columns:
            value = features[column]
            if isinstance(value, bool):
                value = int(value)
            coerced = registry.coerce_value(column, value)
            series[column] = coerced
        return series

    def _coerce_candidate_features(
        self,
        row: pd.Series,
        model_columns: List[str],
        registry: object,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for column in model_columns:
            value = row[column]
            if pd.isna(value):
                value = 0.0
            coerced = registry.coerce_value(column, value)
            result[column] = coerced
        return result

    def _immutable_ok(
        self,
        candidate: Dict[str, object],
        baseline: Dict[str, object],
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
        candidate: Dict[str, object],
        baseline: Dict[str, object],
        mutable_set: set[str],
    ) -> bool:
        for feature, value in candidate.items():
            if feature not in baseline:
                continue
            changed = not self._equal(value, baseline[feature])
            if changed and feature not in mutable_set:
                return False
        return True

    def _medical_ok(self, candidate: Dict[str, object], registry: object) -> bool:
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
        candidate: Dict[str, object],
        baseline: Dict[str, object],
        mutable_allowed: set[str],
        registry: object,
    ) -> bool:
        for feature_name in mutable_allowed:
            if feature_name not in candidate or feature_name not in baseline:
                continue

            feature = registry.get(feature_name)
            if feature is None:
                continue
            direction = getattr(feature, "preferred_direction", "any")
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

    def _build_delta(
        self,
        candidate: Dict[str, object],
        baseline: Dict[str, object],
        mutable_allowed: set[str],
    ) -> Dict[str, float]:
        delta: Dict[str, float] = {}
        for feature in mutable_allowed:
            if feature not in candidate or feature not in baseline:
                continue
            current = float(candidate[feature])
            base = float(baseline[feature])
            diff = current - base
            if abs(diff) < 1e-9:
                continue
            delta[feature] = round(diff, 6)
        return delta

    def _normalized_l1(
        self,
        candidate: Dict[str, object],
        baseline: Dict[str, object],
        registry: object,
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

        if count == 0:
            return 0.0
        return float(total / count)

    def _lof_score(self, candidate_df: pd.DataFrame) -> float:
        assert self.artifacts is not None
        if self.artifacts.lof_model is None:
            return 1.0

        # sklearn score_samples for LOF (novelty=True) is the opposite of LOF score.
        # Convert back so inliers are close to 1.0 and outliers are > 1.0.
        score = float(self.artifacts.lof_model.score_samples(candidate_df)[0])
        return float(max(1e-6, -score))

    def _objective_score(
        self,
        candidate: CounterfactualCandidate,
        preferences: Dict[str, float],
        registry: object,
    ) -> float:
        w_proximity = float(preferences.get("proximity", 0.30))
        w_sparsity = float(preferences.get("sparsity", 0.20))
        w_plausibility = float(preferences.get("plausibility", 0.20))
        w_action_cost = float(preferences.get("action_cost", 0.20))

        changed = max(candidate.metrics.changed_feature_count, 1)
        sparse_score = float(changed)
        proximity_score = candidate.metrics.distance_l1
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

    def _build_prescriptive_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
    ):
        if self.planner is None:
            return None
        try:
            return self.planner.build_plan(request=request, candidate=candidate)
        except Exception:  # noqa: BLE001
            return None

    def _build_stub_candidate(
        self, request: CounterfactualRequest
    ) -> Tuple[CounterfactualCandidate, Dict[str, float]]:
        features = dict(request.instance.features)
        deltas: Dict[str, float] = {}

        for feature_name in request.constraints.mutable_allowed:
            if feature_name not in features:
                continue
            value = features[feature_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            old_value = float(value)
            bound = request.constraints.feature_bounds.get(feature_name)
            new_value = self._bounded_target_value(old_value, bound.min if bound else None)

            if new_value != old_value:
                features[feature_name] = round(new_value, 4)
                deltas[feature_name] = round(new_value - old_value, 4)
                break

        if not deltas:
            raise ValueError("No numeric mutable feature available for stub candidate generation.")

        return (
            CounterfactualCandidate(
                candidate_id="cf_1",
                features=features,
                delta=deltas,
                prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.75),
                metrics=CandidateMetrics(
                    distance_l1=0.1,
                    changed_feature_count=len(deltas),
                    lof_score=1.0,
                    constraint_violations=0,
                ),
            ),
            deltas,
        )

    def _bounded_target_value(self, current_value: float, min_bound: float | None) -> float:
        if min_bound is not None:
            return max(min_bound, current_value - abs(current_value * 0.1))
        return current_value - abs(current_value * 0.1)

    @staticmethod
    def _equal(a: object, b: object) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        return a == b

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)
