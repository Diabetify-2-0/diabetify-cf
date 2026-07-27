from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from diabetify_cf.engine.nn_engine import NearestNeighborCounterfactualEngine
from diabetify_cf.engine.shared import PreparedRequest
from diabetify_cf.schemas import CounterfactualRequest, JSONFeatureValue
from diabetify_cf.verification.fixtures import load_verification_scenarios


@dataclass(frozen=True)
class NNProjectionAblationConfig:
    profile_count: int
    target_class: str
    min_target_probability: float
    mutable_allowed: tuple[str, ...]
    candidate_pool_size: int
    max_neighbors: int
    profile_selection_seed: int
    risk_strata: int = 4

    @classmethod
    def defaults(cls) -> NNProjectionAblationConfig:
        return cls(
            profile_count=50,
            target_class="low_risk",
            min_target_probability=0.5,
            mutable_allowed=(
                "smoking_status",
                "is_cholesterol",
                "moderate_physical_activity_frequency",
                "BMI",
                "is_hypertension",
            ),
            candidate_pool_size=256,
            max_neighbors=64,
            profile_selection_seed=42,
            risk_strata=4,
        ).validated()

    @classmethod
    def from_file(cls, path: str | Path) -> NNProjectionAblationConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            profile_count=int(payload.get("profile_count", 20)),
            target_class=str(payload.get("target_class", "low_risk")),
            min_target_probability=float(payload.get("min_target_probability", 0.5)),
            mutable_allowed=tuple(str(item) for item in payload.get("mutable_allowed", [])),
            candidate_pool_size=int(payload.get("candidate_pool_size", 256)),
            max_neighbors=int(payload.get("max_neighbors", 64)),
            profile_selection_seed=int(payload.get("profile_selection_seed", 42)),
            risk_strata=int(payload.get("risk_strata", 4)),
        ).validated()

    def validated(self) -> NNProjectionAblationConfig:
        if self.profile_count < 1:
            raise ValueError("profile_count must be at least 1")
        if self.target_class != "low_risk":
            raise ValueError("this ablation currently supports target_class='low_risk' only")
        if not 0.0 <= self.min_target_probability <= 1.0:
            raise ValueError("min_target_probability must be between 0 and 1")
        if not self.mutable_allowed:
            raise ValueError("mutable_allowed must contain at least one feature")
        if len(set(self.mutable_allowed)) != len(self.mutable_allowed):
            raise ValueError("mutable_allowed must not contain duplicates")
        if self.candidate_pool_size < 1 or self.max_neighbors < 1:
            raise ValueError("candidate_pool_size and max_neighbors must be at least 1")
        if self.risk_strata < 1:
            raise ValueError("risk_strata must be at least 1")
        return self

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mutable_allowed"] = list(self.mutable_allowed)
        return payload

    def fixed_profile_payload(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "min_target_probability": self.min_target_probability,
            "candidate_pool_size": self.candidate_pool_size,
            "max_neighbors": self.max_neighbors,
        }


@dataclass(frozen=True)
class ProjectionSpec:
    method: str
    features: dict[str, JSONFeatureValue]
    neighbor_rank: int
    prefix_length: int
    ordered_changed_features: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionResult:
    method: str
    features: dict[str, JSONFeatureValue]
    delta: dict[str, float]
    probability_low_risk: float
    proximity_normalized_l1: float
    changed_feature_count: int
    weighted_action_cost: float
    neighbor_rank: int
    prefix_length: int
    ordered_changed_features: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "neighbor_rank": self.neighbor_rank,
            "prefix_length": self.prefix_length,
            "ordered_changed_features": list(self.ordered_changed_features),
            "probability_low_risk": self.probability_low_risk,
            "proximity_normalized_l1": self.proximity_normalized_l1,
            "changed_feature_count": self.changed_feature_count,
            "weighted_action_cost": self.weighted_action_cost,
            "delta": self.delta,
            "features": self.features,
        }


@dataclass(frozen=True)
class ProfileEvaluation:
    reference_index: int | None
    baseline_features: dict[str, JSONFeatureValue]
    baseline_probability_low_risk: float
    mutable_allowed: tuple[str, ...]
    full: ProjectionResult | None
    sparse: ProjectionResult | None

    @property
    def jointly_feasible(self) -> bool:
        return self.full is not None and self.sparse is not None


@dataclass(frozen=True)
class SelectedProfile:
    profile_id: str
    risk_stratum: int | None
    evaluation: ProfileEvaluation


class NNProjectionAblation:
    """Compare full and cheapest-prefix NN projections using only the target gate."""

    def __init__(
        self,
        *,
        engine: NearestNeighborCounterfactualEngine,
        config: NNProjectionAblationConfig,
        profile_input_path: str | Path | None = None,
    ) -> None:
        if engine.artifacts is None:
            detail = engine.initialization_error or "unknown artifact initialization error"
            raise RuntimeError(f"counterfactual engine is not ready: {detail}")

        self.engine = engine
        self.config = config.validated()
        self.profile_input_path = Path(profile_input_path) if profile_input_path else None
        self.artifacts = engine.artifacts
        self.reference_frame = engine._as_model_input_df(
            self.artifacts.reference_data[self.artifacts.feature_columns].copy()
        )
        self.reference_probabilities = np.asarray(
            self.artifacts.model.predict_proba(self.reference_frame)[:, 0],
            dtype=float,
        )
        self.eligible_frame, self.eligible_probabilities = self._build_candidate_pool()

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.profile_input_path is not None:
            selected, selection_summary = self._select_profiles_from_input()
        else:
            selected, selection_summary = self._select_profiles()
        profiles_payload = self._build_profiles_payload(selected, selection_summary)
        report_payload = self._build_report_payload(selected, selection_summary)
        return profiles_payload, report_payload

    def evaluate_reference_profile(self, reference_index: int) -> ProfileEvaluation:
        baseline = self._reference_features(reference_index)
        request = CounterfactualRequest.model_validate(
            {
                "request_id": f"ablation-reference-{reference_index}",
                "target": {
                    "target_class": self.config.target_class,
                    "min_target_probability": self.config.min_target_probability,
                },
                "instance": {"features": baseline},
                "constraints": {"mutable_allowed": list(self.config.mutable_allowed)},
                "generation": {"timeout_ms": 60000},
            }
        )
        prepared = self.engine._prepare_request(request)
        baseline_probability = prepared.base_prediction.probability_low_risk
        if baseline_probability >= self.config.min_target_probability:
            return ProfileEvaluation(
                reference_index=reference_index,
                baseline_features=baseline,
                baseline_probability_low_risk=baseline_probability,
                mutable_allowed=tuple(prepared.mutable_allowed),
                full=None,
                sparse=None,
            )

        ranked_neighbors = self.engine._rank_neighbors(
            eligible=self.eligible_frame,
            eligible_probabilities=self.eligible_probabilities,
            baseline=baseline,
            mutable_allowed=prepared.mutable_allowed,
            prepared=prepared,
        )[: self.config.candidate_pool_size]
        ranked_neighbors = ranked_neighbors[: self.config.max_neighbors]
        specs = self._build_projection_specs(
            ranked_neighbors=ranked_neighbors,
            prepared=prepared,
        )
        results = self._evaluate_projection_specs(specs=specs, prepared=prepared)
        full = self._select_full(results)
        sparse = self._select_sparse_from_same_neighbor(results, full=full)

        evaluation = ProfileEvaluation(
            reference_index=reference_index,
            baseline_features=baseline,
            baseline_probability_low_risk=baseline_probability,
            mutable_allowed=tuple(prepared.mutable_allowed),
            full=full,
            sparse=sparse,
        )
        self._validate_pair(evaluation=evaluation, prepared=prepared)
        return evaluation

    def evaluate_request_profile(self, request: CounterfactualRequest) -> ProfileEvaluation:
        prepared = self.engine._prepare_request(request)
        baseline = dict(prepared.instance_features)
        ranked_neighbors = self.engine._rank_neighbors(
            eligible=self.eligible_frame,
            eligible_probabilities=self.eligible_probabilities,
            baseline=baseline,
            mutable_allowed=prepared.mutable_allowed,
            prepared=prepared,
        )[: self.config.candidate_pool_size]
        ranked_neighbors = ranked_neighbors[: self.config.max_neighbors]
        specs = self._build_projection_specs(
            ranked_neighbors=ranked_neighbors,
            prepared=prepared,
        )
        results = self._evaluate_projection_specs(
            specs=specs,
            prepared=prepared,
            min_target_probability=request.target.min_target_probability,
        )
        full = self._select_full(results)
        sparse = self._select_sparse_from_same_neighbor(results, full=full)

        evaluation = ProfileEvaluation(
            reference_index=None,
            baseline_features=baseline,
            baseline_probability_low_risk=prepared.base_prediction.probability_low_risk,
            mutable_allowed=tuple(prepared.mutable_allowed),
            full=full,
            sparse=sparse,
        )
        self._validate_pair(evaluation=evaluation, prepared=prepared)
        return evaluation

    def _build_candidate_pool(self) -> tuple[pd.DataFrame, np.ndarray]:
        probabilities = self.reference_probabilities
        eligible_mask = probabilities >= self.config.min_target_probability
        eligible = self.reference_frame.loc[eligible_mask].copy()
        eligible_probabilities = probabilities[eligible_mask]

        if eligible.empty:
            indices = np.argsort(probabilities)[::-1][: self.config.candidate_pool_size]
            return self.reference_frame.iloc[indices].copy(), probabilities[indices]
        return eligible, eligible_probabilities

    def _build_projection_specs(
        self,
        *,
        ranked_neighbors: list[pd.Series],
        prepared: PreparedRequest,
    ) -> list[ProjectionSpec]:
        specs: list[ProjectionSpec] = []
        baseline = prepared.instance_features
        for neighbor_rank, neighbor in enumerate(ranked_neighbors, start=1):
            ordered_changes = self.engine._rank_changed_features(
                neighbor=neighbor,
                baseline=baseline,
                mutable_allowed=prepared.mutable_allowed,
                prepared=prepared,
            )
            if not ordered_changes:
                continue

            full_features = dict(baseline)
            for feature_name in ordered_changes:
                full_features[feature_name] = self._coerce_projection_value(
                    feature_name, neighbor[feature_name]
                )
            specs.append(
                ProjectionSpec(
                    method="full",
                    features=full_features,
                    neighbor_rank=neighbor_rank,
                    prefix_length=len(ordered_changes),
                    ordered_changed_features=tuple(ordered_changes),
                )
            )

            for prefix_length in range(1, len(ordered_changes) + 1):
                sparse_features = dict(baseline)
                for feature_name in ordered_changes[:prefix_length]:
                    sparse_features[feature_name] = self._coerce_projection_value(
                        feature_name, neighbor[feature_name]
                    )
                specs.append(
                    ProjectionSpec(
                        method="sparse",
                        features=sparse_features,
                        neighbor_rank=neighbor_rank,
                        prefix_length=prefix_length,
                        ordered_changed_features=tuple(ordered_changes),
                    )
                )
        return specs

    def _evaluate_projection_specs(
        self,
        *,
        specs: list[ProjectionSpec],
        prepared: PreparedRequest,
        min_target_probability: float | None = None,
    ) -> list[ProjectionResult]:
        if not specs:
            return []

        frame = pd.DataFrame(
            [spec.features for spec in specs],
            columns=prepared.model_columns,
        )
        frame = self.engine._as_model_input_df(frame)
        probabilities = np.asarray(self.artifacts.model.predict_proba(frame)[:, 0], dtype=float)
        mutable_set = set(prepared.mutable_allowed)
        results: list[ProjectionResult] = []
        target_probability = (
            self.config.min_target_probability
            if min_target_probability is None
            else min_target_probability
        )

        for spec, probability in zip(specs, probabilities, strict=True):
            if probability < target_probability:
                continue
            raw_delta = self.engine._build_delta(
                spec.features,
                prepared.instance_features,
                mutable_set,
            )
            delta = {
                feature_name: raw_delta[feature_name]
                for feature_name in prepared.mutable_allowed
                if feature_name in raw_delta
            }
            results.append(
                ProjectionResult(
                    method=spec.method,
                    features=spec.features,
                    delta=delta,
                    probability_low_risk=float(probability),
                    proximity_normalized_l1=self.engine._normalized_l1(
                        spec.features,
                        prepared.instance_features,
                        prepared.registry,
                    ),
                    changed_feature_count=len(delta),
                    weighted_action_cost=self._weighted_action_cost(
                        delta=delta,
                        prepared=prepared,
                    ),
                    neighbor_rank=spec.neighbor_rank,
                    prefix_length=spec.prefix_length,
                    ordered_changed_features=spec.ordered_changed_features,
                )
            )
        return results

    @staticmethod
    def _select_full(results: list[ProjectionResult]) -> ProjectionResult | None:
        candidates = [item for item in results if item.method == "full"]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.neighbor_rank,
                item.proximity_normalized_l1,
                -item.probability_low_risk,
            ),
        )

    @staticmethod
    def _select_sparse_from_same_neighbor(
        results: list[ProjectionResult],
        *,
        full: ProjectionResult | None,
    ) -> ProjectionResult | None:
        if full is None:
            return None
        candidates = [
            item
            for item in results
            if item.method == "sparse" and item.neighbor_rank == full.neighbor_rank
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.prefix_length,
                item.proximity_normalized_l1,
                -item.probability_low_risk,
            ),
        )

    def _select_profiles_from_input(self) -> tuple[list[SelectedProfile], dict[str, Any]]:
        assert self.profile_input_path is not None
        scenarios = load_verification_scenarios(self.profile_input_path)
        selected: list[SelectedProfile] = []
        full_feasible_count = 0
        sparse_feasible_count = 0
        jointly_feasible_count = 0
        skipped_profiles: list[dict[str, Any]] = []

        for scenario in scenarios:
            evaluation = self.evaluate_request_profile(scenario.request)
            full_feasible_count += int(evaluation.full is not None)
            sparse_feasible_count += int(evaluation.sparse is not None)
            jointly_feasible_count += int(evaluation.jointly_feasible)
            if not evaluation.jointly_feasible:
                skipped_profiles.append(
                    {
                        "profile_id": scenario.name,
                        "reason": "full_or_sparse_projection_not_target_valid",
                    }
                )
                continue
            selected.append(
                SelectedProfile(
                    profile_id=scenario.name,
                    risk_stratum=None,
                    evaluation=evaluation,
                )
            )

        selection_summary = {
            "input_source": str(self.profile_input_path),
            "input_profile_count": len(scenarios),
            "profiles_evaluated_during_selection": len(scenarios),
            "full_feasible_during_selection": full_feasible_count,
            "sparse_feasible_during_selection": sparse_feasible_count,
            "jointly_feasible_during_selection": jointly_feasible_count,
            "selected_profile_count": len(selected),
            "skipped_profile_count": len(skipped_profiles),
            "skipped_profiles": skipped_profiles,
        }
        return selected, selection_summary

    def _select_profiles(self) -> tuple[list[SelectedProfile], dict[str, Any]]:
        failing_indices = np.flatnonzero(
            self.reference_probabilities < self.config.min_target_probability
        )
        if len(failing_indices) < self.config.profile_count:
            raise RuntimeError(
                "not enough reference profiles below the requested target probability"
            )

        sorted_indices = failing_indices[
            np.argsort(self.reference_probabilities[failing_indices], kind="stable")
        ]
        strata = [part.copy() for part in np.array_split(sorted_indices, self.config.risk_strata)]
        rng = np.random.default_rng(self.config.profile_selection_seed)
        for part in strata:
            rng.shuffle(part)

        base_quota, remainder = divmod(self.config.profile_count, len(strata))
        quotas = [base_quota + (1 if index < remainder else 0) for index in range(len(strata))]
        selected: list[SelectedProfile] = []
        evaluated_indices: set[int] = set()
        evaluated_count = 0
        full_feasible_count = 0
        sparse_feasible_count = 0
        jointly_feasible_count = 0

        for stratum_index, (part, quota) in enumerate(zip(strata, quotas, strict=True), start=1):
            accepted = 0
            for raw_index in part:
                reference_index = int(raw_index)
                evaluated_indices.add(reference_index)
                evaluation = self.evaluate_reference_profile(reference_index)
                evaluated_count += 1
                full_feasible_count += int(evaluation.full is not None)
                sparse_feasible_count += int(evaluation.sparse is not None)
                jointly_feasible_count += int(evaluation.jointly_feasible)
                if not evaluation.jointly_feasible:
                    continue
                selected.append(
                    SelectedProfile(
                        profile_id=f"reference_profile_{reference_index}",
                        risk_stratum=stratum_index,
                        evaluation=evaluation,
                    )
                )
                accepted += 1
                if accepted >= quota:
                    break

        if len(selected) < self.config.profile_count:
            remaining = np.asarray(
                [index for index in failing_indices if int(index) not in evaluated_indices],
                dtype=int,
            )
            rng.shuffle(remaining)
            for raw_index in remaining:
                reference_index = int(raw_index)
                evaluation = self.evaluate_reference_profile(reference_index)
                evaluated_count += 1
                full_feasible_count += int(evaluation.full is not None)
                sparse_feasible_count += int(evaluation.sparse is not None)
                jointly_feasible_count += int(evaluation.jointly_feasible)
                if not evaluation.jointly_feasible:
                    continue
                selected.append(
                    SelectedProfile(
                        profile_id=f"reference_profile_{reference_index}",
                        risk_stratum=0,
                        evaluation=evaluation,
                    )
                )
                if len(selected) >= self.config.profile_count:
                    break

        if len(selected) != self.config.profile_count:
            raise RuntimeError(
                f"required {self.config.profile_count} jointly feasible profiles, "
                f"but found {len(selected)}"
            )

        selection_summary = {
            "total_reference_profiles": len(self.reference_frame),
            "baseline_profiles_below_target": len(failing_indices),
            "profiles_evaluated_during_selection": evaluated_count,
            "full_feasible_during_selection": full_feasible_count,
            "sparse_feasible_during_selection": sparse_feasible_count,
            "jointly_feasible_during_selection": jointly_feasible_count,
            "selected_profile_count": len(selected),
            "risk_strata": len(strata),
            "stratum_quotas": quotas,
        }
        return selected, selection_summary

    def _build_profiles_payload(
        self,
        selected: list[SelectedProfile],
        selection_summary: dict[str, Any],
    ) -> dict[str, Any]:
        selection_policy = (
            "Use fixed scenario profiles from profile_input_path; retain only profiles for "
            "which both methods satisfy the target gate."
            if self.profile_input_path is not None
            else (
                "Deterministic risk-stratified sampling from reference profiles below the "
                "target; retain only profiles for which both methods satisfy the target gate."
            )
        )
        return {
            "experiment": "nn_full_vs_prefix_sparse_target_gate_only",
            "selection_policy": selection_policy,
            "config": self._config_payload(),
            "selection_summary": selection_summary,
            "profiles": [_selected_profile_payload(item) for item in selected],
        }

    def _build_report_payload(
        self,
        selected: list[SelectedProfile],
        selection_summary: dict[str, Any],
    ) -> dict[str, Any]:
        comparisons = []
        full_proximities: list[float] = []
        sparse_proximities: list[float] = []
        full_sparsities: list[int] = []
        sparse_sparsities: list[int] = []

        for item in selected:
            full = item.evaluation.full
            sparse = item.evaluation.sparse
            assert full is not None and sparse is not None
            proximity_delta = sparse.proximity_normalized_l1 - full.proximity_normalized_l1
            sparsity_reduction = full.changed_feature_count - sparse.changed_feature_count
            full_proximities.append(full.proximity_normalized_l1)
            sparse_proximities.append(sparse.proximity_normalized_l1)
            full_sparsities.append(full.changed_feature_count)
            sparse_sparsities.append(sparse.changed_feature_count)
            comparison_payload: dict[str, Any] = {
                "profile_id": item.profile_id,
                "baseline": {
                    "probability_low_risk": (item.evaluation.baseline_probability_low_risk),
                    "mutable_allowed": list(item.evaluation.mutable_allowed),
                    "features": item.evaluation.baseline_features,
                },
                "full_projection": full.to_payload(),
                "prefix_sparse_projection": sparse.to_payload(),
                "comparison": {
                    "proximity_delta_sparse_minus_full": proximity_delta,
                    "sparsity_reduction_full_minus_sparse": sparsity_reduction,
                    "proximity_winner": _lower_is_better_winner(
                        sparse.proximity_normalized_l1,
                        full.proximity_normalized_l1,
                    ),
                    "sparsity_winner": _lower_is_better_winner(
                        float(sparse.changed_feature_count),
                        float(full.changed_feature_count),
                    ),
                },
            }
            if item.evaluation.reference_index is not None:
                comparison_payload["reference_index"] = item.evaluation.reference_index
            if item.risk_stratum is not None:
                comparison_payload["risk_stratum"] = item.risk_stratum
            comparisons.append(comparison_payload)

        full_mean_proximity = mean(full_proximities)
        sparse_mean_proximity = mean(sparse_proximities)
        full_mean_sparsity = mean(full_sparsities)
        sparse_mean_sparsity = mean(sparse_sparsities)
        sparsity_reductions = [
            full - sparse for full, sparse in zip(full_sparsities, sparse_sparsities, strict=True)
        ]
        return {
            "experiment": "nn_full_vs_prefix_sparse_target_gate_only",
            "methodology": {
                "shared_neighbor_ranking": True,
                "same_source_neighbor": True,
                "full_selection": "first target-valid full projection by neighbor rank",
                "sparse_selection": (
                    "smallest target-valid cheapest-feature prefix from the same neighbor "
                    "selected by the full method"
                ),
                "active_selection_checks": ["target_probability"],
                "omitted_selection_checks": [
                    "directional_constraint",
                    "allowed_transition",
                    "medical_range",
                    "final_objective_ranking",
                ],
                "primary_proximity_metric": "heom",
                "primary_sparsity_metric": "changed_feature_count",
            },
            "config": self._config_payload(),
            "selection_summary": selection_summary,
            "summary": {
                "valid_pair_count": len(selected),
                "full_mean_proximity": full_mean_proximity,
                "sparse_mean_proximity": sparse_mean_proximity,
                "full_mean_changed_feature_count": full_mean_sparsity,
                "sparse_mean_changed_feature_count": sparse_mean_sparsity,
                "proximity_reduction_percent": (
                    (full_mean_proximity - sparse_mean_proximity)
                    / full_mean_proximity
                    * 100.0
                    if full_mean_proximity > 0
                    else 0.0
                ),
                "changed_feature_count_reduction_percent": (
                    mean(sparsity_reductions) / full_mean_sparsity * 100.0
                    if full_mean_sparsity > 0
                    else 0.0
                ),
            },
            "profiles": comparisons,
        }

    def _validate_pair(
        self,
        *,
        evaluation: ProfileEvaluation,
        prepared: PreparedRequest,
    ) -> None:
        for candidate in (evaluation.full, evaluation.sparse):
            if candidate is None:
                continue
            if candidate.probability_low_risk < self.config.min_target_probability:
                raise RuntimeError("selected ablation candidate does not satisfy target gate")
            if not self.engine._mutable_ok(
                candidate.features,
                prepared.instance_features,
                set(prepared.mutable_allowed),
            ):
                raise RuntimeError("selected ablation candidate changes a non-mutable feature")
            if set(candidate.features) != set(prepared.model_columns):
                raise RuntimeError(
                    "selected ablation candidate does not contain all model features"
                )
        if (
            evaluation.full is not None
            and evaluation.sparse is not None
            and evaluation.full.neighbor_rank != evaluation.sparse.neighbor_rank
        ):
            raise RuntimeError("full and sparse candidates must use the same source neighbor")

    def _reference_features(self, reference_index: int) -> dict[str, JSONFeatureValue]:
        if reference_index < 0 or reference_index >= len(self.reference_frame):
            raise IndexError(f"reference index out of range: {reference_index}")
        row = self.reference_frame.iloc[reference_index]
        return {
            column: self._coerce_projection_value(column, row[column])
            for column in self.artifacts.feature_columns
        }

    def _coerce_projection_value(self, feature_name: str, value: object) -> JSONFeatureValue:
        coerced = self.artifacts.feature_registry.coerce_value(feature_name, value)
        if isinstance(coerced, np.generic):
            coerced = coerced.item()
        if not isinstance(coerced, (int, float, bool, str)):
            coerced = float(str(coerced))
        return coerced

    def _weighted_action_cost(
        self,
        *,
        delta: dict[str, float],
        prepared: PreparedRequest,
    ) -> float:
        total = 0.0
        for feature_name, value in delta.items():
            total += self.engine._weighted_normalized_delta(
                feature_name=feature_name,
                delta=abs(float(value)),
                prepared=prepared,
            )
        return total

    def _config_payload(self) -> dict[str, Any]:
        if self.profile_input_path is not None:
            return self.config.fixed_profile_payload()
        return self.config.to_payload()


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_profile_payload(item: SelectedProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_id": item.profile_id,
        "baseline_probability_low_risk": item.evaluation.baseline_probability_low_risk,
        "mutable_allowed": list(item.evaluation.mutable_allowed),
        "features": item.evaluation.baseline_features,
    }
    if item.evaluation.reference_index is not None:
        payload["reference_index"] = item.evaluation.reference_index
    if item.risk_stratum is not None:
        payload["risk_stratum"] = item.risk_stratum
    return payload


def _lower_is_better_winner(sparse: float, full: float) -> str:
    if abs(sparse - full) < 1e-12:
        return "tie"
    return "prefix_sparse" if sparse < full else "full"
