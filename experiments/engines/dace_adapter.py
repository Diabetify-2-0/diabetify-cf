from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pulp
from sklearn.ensemble import RandomForestClassifier

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
class DaceSolverOptions:
    surrogate_n_estimators: int = 64
    surrogate_max_depth: int = 6
    max_changed_features: int = 3
    max_candidates_per_feature: int = 24
    threshold_epsilon: float = 1e-4
    solver: str = "PULP_CBC_CMD"
    relative_gap: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> DaceSolverOptions:
        raw = (config or {}).get("engine_options") or {}
        if not isinstance(raw, dict):
            raise ValueError("engine_options must be an object.")

        surrogate_n_estimators = int(raw.get("surrogate_n_estimators", cls.surrogate_n_estimators))
        if surrogate_n_estimators < 1:
            raise ValueError("DACE option 'surrogate_n_estimators' must be >= 1.")

        surrogate_max_depth = int(raw.get("surrogate_max_depth", cls.surrogate_max_depth))
        if surrogate_max_depth < 1:
            raise ValueError("DACE option 'surrogate_max_depth' must be >= 1.")

        max_changed_features = int(raw.get("max_changed_features", cls.max_changed_features))
        if max_changed_features < 1:
            raise ValueError("DACE option 'max_changed_features' must be >= 1.")

        max_candidates_per_feature = int(
            raw.get("max_candidates_per_feature", cls.max_candidates_per_feature)
        )
        if max_candidates_per_feature < 2:
            raise ValueError("DACE option 'max_candidates_per_feature' must be >= 2.")

        threshold_epsilon = float(raw.get("threshold_epsilon", cls.threshold_epsilon))
        if threshold_epsilon <= 0.0:
            raise ValueError("DACE option 'threshold_epsilon' must be > 0.")

        solver = str(raw.get("solver", cls.solver)).strip() or cls.solver
        relative_gap_raw = raw.get("relative_gap")
        relative_gap = None if relative_gap_raw is None else float(relative_gap_raw)
        if relative_gap is not None and relative_gap < 0.0:
            raise ValueError("DACE option 'relative_gap' must be >= 0.")

        return cls(
            surrogate_n_estimators=surrogate_n_estimators,
            surrogate_max_depth=surrogate_max_depth,
            max_changed_features=max_changed_features,
            max_candidates_per_feature=max_candidates_per_feature,
            threshold_epsilon=threshold_epsilon,
            solver=solver,
            relative_gap=relative_gap,
        )


@dataclass(frozen=True)
class _LeafRegion:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    low_risk_probability: float


class DaceCandidateGenerator:
    """MILP adapter inspired by DACE forest counterfactuals on a surrogate RF."""

    engine_version = "dace_rf_surrogate_experiment_v1"

    def __init__(
        self,
        model_path: str,
        columns_path: str,
        reference_data_path: str,
        feature_registry_path: str,
        options: DaceSolverOptions | None = None,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        self.options = options or DaceSolverOptions()
        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
            self._surrogate_forest = self._fit_surrogate_forest()
            self._forest_regions = self._build_forest_regions()
        except Exception as exc:
            self.artifacts = None
            self.initialization_error = str(exc)
            self._surrogate_forest = None
            self._forest_regions = []

    def generate_raw(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        assert self._surrogate_forest is not None

        action_values = self._build_action_values(prepared)
        if not action_values:
            return pd.DataFrame()

        coefficients = self._cost_matrix(prepared)
        candidates: list[dict[str, Any]] = []
        exclusion_signatures: list[dict[str, int]] = []

        for _ in range(request.generation.total_cfs):
            signature = self._solve_once(
                request=request,
                prepared=prepared,
                action_values=action_values,
                coefficients=coefficients,
                excluded_signatures=exclusion_signatures,
            )
            if signature is None:
                break
            exclusion_signatures.append(signature)
            state = dict(prepared.instance_features)
            for feature_name, index in signature.items():
                state[feature_name] = self._coerce_feature_value(
                    feature_name=feature_name,
                    value=action_values[feature_name][index],
                )
            candidates.append(state)

        if not candidates:
            return pd.DataFrame()

        raw = pd.DataFrame(candidates).drop_duplicates()
        ordered = raw[self.artifacts.feature_columns]
        return postprocessor.as_model_input_df(ordered)

    def _fit_surrogate_forest(self) -> RandomForestClassifier:
        assert self.artifacts is not None
        frame = self.artifacts.reference_data[self.artifacts.feature_columns]
        labels = np.asarray(self.artifacts.model.predict(frame), dtype=int)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            raise ValueError("DACE surrogate forest requires at least two predicted classes.")

        forest = RandomForestClassifier(
            n_estimators=self.options.surrogate_n_estimators,
            max_depth=self.options.surrogate_max_depth,
            random_state=42,
        )
        forest.fit(frame, labels)
        return forest

    def _build_forest_regions(self) -> list[list[_LeafRegion]]:
        assert self.artifacts is not None
        assert self._surrogate_forest is not None

        regions: list[list[_LeafRegion]] = []
        feature_count = len(self.artifacts.feature_columns)
        class_to_low_risk_index = {
            int(label): index for index, label in enumerate(self._surrogate_forest.classes_)
        }
        low_risk_index = class_to_low_risk_index.get(0)
        if low_risk_index is None:
            raise ValueError("DACE surrogate forest must expose class '0' as low risk.")

        for estimator in self._surrogate_forest.estimators_:
            tree = estimator.tree_
            tree_regions: list[_LeafRegion] = []

            self._walk_tree_regions(
                tree=tree,
                node_id=0,
                lower=[-math.inf] * feature_count,
                upper=[math.inf] * feature_count,
                low_risk_index=low_risk_index,
                out=tree_regions,
            )
            regions.append(tree_regions)
        return regions

    def _walk_tree_regions(
        self,
        *,
        tree: Any,
        node_id: int,
        lower: list[float],
        upper: list[float],
        low_risk_index: int,
        out: list[_LeafRegion],
    ) -> None:
        feature_index = int(tree.feature[node_id])
        if feature_index == -2:
            values = tree.value[node_id][0]
            total = float(np.sum(values))
            if total <= 0.0:
                low_risk_probability = 0.0
            else:
                low_risk_probability = float(values[low_risk_index] / total)
            out.append(
                _LeafRegion(
                    lower=tuple(lower),
                    upper=tuple(upper),
                    low_risk_probability=low_risk_probability,
                )
            )
            return

        threshold = float(tree.threshold[node_id])
        left_upper = list(upper)
        left_upper[feature_index] = min(left_upper[feature_index], threshold)
        self._walk_tree_regions(
            tree=tree,
            node_id=int(tree.children_left[node_id]),
            lower=list(lower),
            upper=left_upper,
            low_risk_index=low_risk_index,
            out=out,
        )

        right_lower = list(lower)
        right_lower[feature_index] = max(right_lower[feature_index], threshold)
        self._walk_tree_regions(
            tree=tree,
            node_id=int(tree.children_right[node_id]),
            lower=right_lower,
            upper=list(upper),
            low_risk_index=low_risk_index,
            out=out,
        )

    def _build_action_values(
        self,
        prepared: PreparedExperimentRequest,
    ) -> dict[str, list[float]]:
        assert self.artifacts is not None
        action_values: dict[str, list[float]] = {}
        for feature_name in prepared.mutable_allowed:
            feature = self.artifacts.feature_registry.get(feature_name)
            lower, upper = self._bounds_for_feature(feature_name, feature, prepared)
            baseline = float(prepared.instance_features[feature_name])
            values = self._feature_action_values(
                feature_name=feature_name,
                feature=feature,
                baseline=baseline,
                lower=lower,
                upper=upper,
            )
            if values:
                action_values[feature_name] = values
        return action_values

    def _feature_action_values(
        self,
        *,
        feature_name: str,
        feature: FeatureDefinition | None,
        baseline: float,
        lower: float,
        upper: float,
    ) -> list[float]:
        assert self.artifacts is not None
        candidates: list[float] = [baseline]

        if feature is not None and feature.is_binary:
            candidates.extend([lower, upper])
        else:
            thresholds = self._feature_thresholds(feature_name, lower=lower, upper=upper)
            reference_values = self._reference_values(feature_name, lower=lower, upper=upper)
            candidates.extend(reference_values)
            for threshold in thresholds:
                candidates.append(threshold - self.options.threshold_epsilon)
                candidates.append(threshold + self.options.threshold_epsilon)
            candidates.extend([lower, upper])

        deduped: list[float] = []
        seen: set[float] = set()
        for value in candidates:
            clipped = round(float(min(max(value, lower), upper)), 6)
            if clipped in seen:
                continue
            deduped.append(clipped)
            seen.add(clipped)

        deduped.sort(key=lambda value: (abs(value - baseline), value))
        return deduped[: self.options.max_candidates_per_feature]

    def _feature_thresholds(
        self,
        feature_name: str,
        *,
        lower: float,
        upper: float,
    ) -> list[float]:
        assert self._surrogate_forest is not None
        assert self.artifacts is not None
        feature_index = self.artifacts.feature_columns.index(feature_name)
        values: set[float] = set()
        for estimator in self._surrogate_forest.estimators_:
            tree = estimator.tree_
            mask = tree.feature == feature_index
            for threshold in tree.threshold[mask]:
                threshold_value = float(threshold)
                if lower <= threshold_value <= upper:
                    values.add(threshold_value)
        return sorted(values)

    def _reference_values(
        self,
        feature_name: str,
        *,
        lower: float,
        upper: float,
    ) -> list[float]:
        assert self.artifacts is not None
        series = pd.to_numeric(
            self.artifacts.reference_data[feature_name],
            errors="coerce",
        ).dropna()
        clipped = series[(series >= lower) & (series <= upper)]
        if clipped.empty:
            return []
        unique_values = sorted(float(value) for value in clipped.unique())
        if len(unique_values) <= self.options.max_candidates_per_feature:
            return unique_values
        quantiles = np.linspace(0.0, 1.0, num=self.options.max_candidates_per_feature)
        return [float(value) for value in clipped.quantile(quantiles).tolist()]

    def _cost_matrix(self, prepared: PreparedExperimentRequest) -> np.ndarray:
        assert self.artifacts is not None
        baseline_frame = self.artifacts.reference_data[self.artifacts.feature_columns]
        base_prediction = prepared.base_prediction.class_name
        if hasattr(self.artifacts.model, "predict"):
            labels = np.asarray(self.artifacts.model.predict(baseline_frame), dtype=int)
            class_label = 0 if base_prediction == "low_risk" else 1
            class_rows = baseline_frame.loc[labels == class_label]
            if len(class_rows) >= 2:
                baseline_frame = class_rows
        covariance = np.cov(baseline_frame.to_numpy(dtype=float), rowvar=False)
        if covariance.ndim == 0:
            covariance = np.eye(len(self.artifacts.feature_columns), dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        covariance += 1e-6 * np.eye(covariance.shape[0], dtype=float)
        inv_covariance = np.linalg.pinv(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(inv_covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-9)
        sqrt_inv_covariance = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        return np.asarray(sqrt_inv_covariance, dtype=float)

    def _solve_once(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        action_values: dict[str, list[float]],
        coefficients: np.ndarray,
        excluded_signatures: list[dict[str, int]],
    ) -> dict[str, int] | None:
        assert self.artifacts is not None
        feature_names = list(action_values)
        feature_indices = {
            feature_name: self.artifacts.feature_columns.index(feature_name)
            for feature_name in feature_names
        }

        probability_threshold = self._target_probability_threshold(request)
        problem = pulp.LpProblem("dace_surrogate", pulp.LpMinimize)
        act_vars = {
            feature_name: pulp.LpVariable(
                f"act_{feature_name}",
                lowBound=min(action_values[feature_name]),
                upBound=max(action_values[feature_name]),
            )
            for feature_name in feature_names
        }
        choice_vars = {
            feature_name: [
                pulp.LpVariable(f"pi_{feature_name}_{index}", cat="Binary")
                for index in range(len(action_values[feature_name]))
            ]
            for feature_name in feature_names
        }
        nonzero_vars = {
            feature_name: [
                0 if abs(value - float(prepared.instance_features[feature_name])) < 1e-9 else 1
                for value in action_values[feature_name]
            ]
            for feature_name in feature_names
        }
        tree_vars = [
            [pulp.LpVariable(f"phi_{tree_index}_{leaf_index}", cat="Binary")
             for leaf_index in range(len(tree_regions))]
            for tree_index, tree_regions in enumerate(self._forest_regions)
        ]
        distance_vars = [
            pulp.LpVariable(f"dist_{index}", lowBound=0.0)
            for index in range(len(self.artifacts.feature_columns))
        ]

        for feature_name in feature_names:
            problem += pulp.lpSum(choice_vars[feature_name]) == 1
            problem += (
                act_vars[feature_name]
                == pulp.lpSum(
                    value * choice_vars[feature_name][index]
                    for index, value in enumerate(action_values[feature_name])
                )
            )

        problem += pulp.lpSum(
            nonzero_vars[feature_name][index] * choice_vars[feature_name][index]
            for feature_name in feature_names
            for index in range(len(action_values[feature_name]))
        ) <= min(self.options.max_changed_features, len(feature_names))

        for row_index in range(coefficients.shape[0]):
            linear_expr = []
            for feature_name in feature_names:
                feature_index = feature_indices[feature_name]
                baseline = float(prepared.instance_features[feature_name])
                linear_expr.append(
                    coefficients[row_index, feature_index] * (act_vars[feature_name] - baseline)
                )
            problem += distance_vars[row_index] >= pulp.lpSum(linear_expr)
            problem += distance_vars[row_index] >= -pulp.lpSum(linear_expr)

        objective_weights = {
            feature_name: (
                prepared.registry.get(feature_name).cost_weight
                if prepared.registry.get(feature_name) is not None
                else 1.0
            )
            for feature_name in feature_names
        }
        problem += pulp.lpSum(distance_vars) + 1e-3 * pulp.lpSum(
            objective_weights[feature_name]
            * abs(
                action_values[feature_name][index]
                - float(prepared.instance_features[feature_name])
            )
            * choice_vars[feature_name][index]
            for feature_name in feature_names
            for index in range(len(action_values[feature_name]))
        )

        leaf_probability_terms = []
        for tree_index, tree_regions in enumerate(self._forest_regions):
            problem += pulp.lpSum(tree_vars[tree_index]) == 1
            for leaf_index, region in enumerate(tree_regions):
                valid_terms = []
                for feature_name in feature_names:
                    feature_index = feature_indices[feature_name]
                    lower = region.lower[feature_index]
                    upper = region.upper[feature_index]
                    valid_terms.extend(
                        choice_vars[feature_name][index]
                        for index, value in enumerate(action_values[feature_name])
                        if lower < value <= upper
                    )
                if valid_terms:
                    problem += (
                        len(feature_names) * tree_vars[tree_index][leaf_index]
                        <= pulp.lpSum(valid_terms)
                    )
                else:
                    problem += tree_vars[tree_index][leaf_index] == 0
                leaf_probability_terms.append(
                    region.low_risk_probability * tree_vars[tree_index][leaf_index]
                )

        mean_low_risk_probability = pulp.lpSum(leaf_probability_terms) / max(
            len(self._forest_regions), 1
        )
        problem += mean_low_risk_probability >= probability_threshold + 1e-8

        for excluded in excluded_signatures:
            problem += pulp.lpSum(
                choice_vars[feature_name][index]
                for feature_name, index in excluded.items()
            ) <= max(len(excluded) - 1, 0)

        solver = pulp.getSolver(
            self.options.solver,
            msg=False,
            timeLimit=max(1, math.ceil(request.generation.timeout_ms / 1000)),
            gapRel=self.options.relative_gap,
        )
        status = problem.solve(solver)
        if pulp.LpStatus.get(status) != "Optimal":
            feasible_statuses = {"Optimal", "Integer Feasible", "Not Solved"}
            if pulp.LpStatus.get(status) not in feasible_statuses:
                return None
            if problem.status not in {1, 0}:
                return None

        signature: dict[str, int] = {}
        for feature_name in feature_names:
            chosen_index = next(
                (
                    index
                    for index, variable in enumerate(choice_vars[feature_name])
                    if variable.value() is not None and round(float(variable.value())) == 1
                ),
                None,
            )
            if chosen_index is None:
                return None
            signature[feature_name] = chosen_index
        return signature

    def _bounds_for_feature(
        self,
        feature_name: str,
        feature: FeatureDefinition | None,
        prepared: PreparedExperimentRequest,
    ) -> tuple[float, float]:
        if feature_name in prepared.permitted_range:
            lower, upper = prepared.permitted_range[feature_name]
            return float(lower), float(upper)
        assert self.artifacts is not None
        if (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            return float(feature.global_min), float(feature.global_max)
        series = self.artifacts.reference_data[feature_name]
        return float(series.min()), float(series.max())

    def _coerce_feature_value(self, feature_name: str, value: float) -> Any:
        assert self.artifacts is not None
        return self.artifacts.feature_registry.coerce_value(feature_name, value)

    @staticmethod
    def _target_probability_threshold(request: CounterfactualRequest) -> float:
        target_class = request.target.target_class.strip().lower()
        if target_class in {"low", "low_risk", "non_diabetes", "0"}:
            return float(request.target.min_target_probability)
        raise ValueError("Current DACE surrogate adapter only supports low-risk targets.")


class DaceExperimentAdapter(ExperimentEngine):
    name = "dace"

    def __init__(
        self,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.engine = DaceCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
            options=DaceSolverOptions.from_config(config),
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
            message = "Experiment DACE generator is not fully configured yet."
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
