from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.reason_codes import Status
from diabetify_cf.schemas import CounterfactualRequest, JSONFeatureValue
from diabetify_cf.verification.fixtures import load_verification_scenarios
from diabetify_cf.verification.runner import VerificationScenario


@dataclass(frozen=True)
class NNCECandidateResult:
    candidate_features: dict[str, JSONFeatureValue] | None
    source_reference_index: int | None
    distance: float | None
    probability_low_risk: float | None
    target_satisfied: bool
    changed_immutable_features: tuple[str, ...]
    changed_outside_selected_mutable_features: tuple[str, ...]

    @property
    def immutable_violation(self) -> bool:
        return bool(self.changed_immutable_features)

    @property
    def outside_selected_mutable_violation(self) -> bool:
        return bool(self.changed_outside_selected_mutable_features)


@dataclass(frozen=True)
class NNCEActionabilityScenarioResult:
    scenario_name: str
    request_id: str
    mutable_allowed: tuple[str, ...]
    pure_nnce: NNCECandidateResult
    adapted_nnce: NNCECandidateResult


class NNCEActionabilityBenchmark:
    """Compare pure NNCE against an actionability-adapted full projection variant."""

    def __init__(
        self,
        *,
        artifacts: ModelArtifacts,
        scenarios: list[VerificationScenario],
    ) -> None:
        self.artifacts = artifacts
        self.scenarios = scenarios
        self.registry = artifacts.feature_registry
        self.feature_columns = artifacts.feature_columns
        self.reference_frame = self._as_model_input_df(
            artifacts.reference_data[self.feature_columns].copy()
        )
        self.reference_probabilities = np.asarray(
            artifacts.model.predict_proba(self.reference_frame)[:, 0],
            dtype=float,
        )

    @classmethod
    def from_fixture_path(
        cls,
        *,
        artifacts: ModelArtifacts,
        scenarios_path: str | Path,
    ) -> NNCEActionabilityBenchmark:
        scenarios = load_verification_scenarios(
            scenarios_path,
            include_tags=("actionability_profile",),
        )
        return cls(artifacts=artifacts, scenarios=scenarios)

    def run(self) -> dict[str, Any]:
        scenario_results = [self.evaluate_scenario(scenario) for scenario in self.scenarios]
        return {
            "experiment": "pure_nnce_vs_actionability_adapted_nnce",
            "methodology": {
                "pure_nnce": (
                    "Nearest eligible reference record using HEOM over all model features."
                ),
                "adapted_nnce": (
                    "Nearest eligible reference record using HEOM over user-selected mutable "
                    "features, followed by full projection of those selected features onto the "
                    "baseline instance."
                ),
                "distance": (
                    "HEOM: normalized numeric distance for continuous/ordinal features, "
                    "overlap distance for binary features, aggregated with Euclidean norm."
                ),
                "target_candidate_pool": (
                    "Reference records whose low-risk probability satisfies the scenario target."
                ),
            },
            "summary": self._build_summary(scenario_results),
            "scenarios": [self._scenario_payload(result) for result in scenario_results],
        }

    def evaluate_scenario(
        self,
        scenario: VerificationScenario,
    ) -> NNCEActionabilityScenarioResult:
        request = scenario.request
        baseline = self._canonical_baseline(request)
        mutable_allowed = tuple(
            self.registry.canonicalize_feature_names(request.constraints.mutable_allowed)
        )

        pure = self._pure_nnce(request=request, baseline=baseline)
        adapted = self._adapted_nnce(
            request=request,
            baseline=baseline,
            mutable_allowed=mutable_allowed,
        )
        return NNCEActionabilityScenarioResult(
            scenario_name=scenario.name,
            request_id=request.request_id,
            mutable_allowed=mutable_allowed,
            pure_nnce=pure,
            adapted_nnce=adapted,
        )

    def _pure_nnce(
        self,
        *,
        request: CounterfactualRequest,
        baseline: dict[str, JSONFeatureValue],
    ) -> NNCECandidateResult:
        neighbor = self._nearest_eligible_neighbor(
            request=request,
            baseline=baseline,
            distance_features=tuple(self.feature_columns),
        )
        if neighbor is None:
            return self._empty_candidate_result()

        reference_index, distance = neighbor
        candidate = self._reference_features(reference_index)
        return self._build_candidate_result(
            request=request,
            baseline=baseline,
            candidate=candidate,
            source_reference_index=reference_index,
            distance=distance,
        )

    def _adapted_nnce(
        self,
        *,
        request: CounterfactualRequest,
        baseline: dict[str, JSONFeatureValue],
        mutable_allowed: tuple[str, ...],
    ) -> NNCECandidateResult:
        if not mutable_allowed:
            return self._build_candidate_result(
                request=request,
                baseline=baseline,
                candidate=dict(baseline),
                source_reference_index=None,
                distance=None,
            )

        neighbor = self._nearest_projected_target_neighbor(
            request=request,
            baseline=baseline,
            distance_features=mutable_allowed,
            projection_features=mutable_allowed,
        )
        if neighbor is None:
            return self._empty_candidate_result()

        reference_index, distance, projected = neighbor
        return self._build_candidate_result(
            request=request,
            baseline=baseline,
            candidate=projected,
            source_reference_index=reference_index,
            distance=distance,
        )

    def _nearest_eligible_neighbor(
        self,
        *,
        request: CounterfactualRequest,
        baseline: dict[str, JSONFeatureValue],
        distance_features: tuple[str, ...],
    ) -> tuple[int, float] | None:
        eligible_indices = np.flatnonzero(
            self.reference_probabilities >= request.target.min_target_probability
        )
        if len(eligible_indices) == 0:
            return None

        best: tuple[int, float] | None = None
        for raw_index in eligible_indices:
            reference_index = int(raw_index)
            row = self.reference_frame.iloc[reference_index]
            candidate = {
                feature_name: self._coerce_feature_value(feature_name, row[feature_name])
                for feature_name in self.feature_columns
            }
            distance = heom_distance(
                baseline=baseline,
                candidate=candidate,
                feature_names=distance_features,
                registry=self.registry,
            )
            if best is None or (distance, reference_index) < (best[1], best[0]):
                best = (reference_index, distance)
        return best

    def _nearest_projected_target_neighbor(
        self,
        *,
        request: CounterfactualRequest,
        baseline: dict[str, JSONFeatureValue],
        distance_features: tuple[str, ...],
        projection_features: tuple[str, ...],
    ) -> tuple[int, float, dict[str, JSONFeatureValue]] | None:
        projected_frame = self.reference_frame.copy()
        projection_set = set(projection_features)
        for feature_name in self.feature_columns:
            if feature_name not in projection_set:
                projected_frame[feature_name] = baseline[feature_name]

        projected_frame = self._as_model_input_df(projected_frame)
        projected_probabilities = np.asarray(
            self.artifacts.model.predict_proba(projected_frame)[:, 0],
            dtype=float,
        )
        target_valid_indices = np.flatnonzero(
            projected_probabilities >= request.target.min_target_probability
        )
        if len(target_valid_indices) == 0:
            return None

        best: tuple[int, float, dict[str, JSONFeatureValue]] | None = None
        for raw_index in target_valid_indices:
            reference_index = int(raw_index)
            reference_candidate = self._reference_features(reference_index)
            projected = dict(baseline)
            for feature_name in projection_features:
                projected[feature_name] = reference_candidate[feature_name]

            distance = heom_distance(
                baseline=baseline,
                candidate=reference_candidate,
                feature_names=distance_features,
                registry=self.registry,
            )
            if best is None or (distance, reference_index) < (best[1], best[0]):
                best = (reference_index, distance, projected)
        return best

    def _build_candidate_result(
        self,
        *,
        request: CounterfactualRequest,
        baseline: dict[str, JSONFeatureValue],
        candidate: dict[str, JSONFeatureValue],
        source_reference_index: int | None,
        distance: float | None,
    ) -> NNCECandidateResult:
        probability_low_risk = self._predict_low_risk_probability(candidate)
        target_satisfied = probability_low_risk >= request.target.min_target_probability
        mutable_allowed = set(
            self.registry.canonicalize_feature_names(request.constraints.mutable_allowed)
        )
        immutable_features = set(self.registry.immutable_defaults())
        outside_selected_mutable = {
            feature.name
            for feature in self.registry.features
            if feature.actionable and feature.name not in mutable_allowed
        }
        changed_immutable = tuple(
            feature_name
            for feature_name in self.feature_columns
            if feature_name in immutable_features
            and _changed(candidate[feature_name], baseline[feature_name])
        )
        changed_outside_selected_mutable = tuple(
            feature_name
            for feature_name in self.feature_columns
            if feature_name in outside_selected_mutable
            and _changed(candidate[feature_name], baseline[feature_name])
        )
        return NNCECandidateResult(
            candidate_features=candidate,
            source_reference_index=source_reference_index,
            distance=distance,
            probability_low_risk=probability_low_risk,
            target_satisfied=target_satisfied,
            changed_immutable_features=changed_immutable,
            changed_outside_selected_mutable_features=changed_outside_selected_mutable,
        )

    def _predict_low_risk_probability(
        self,
        candidate: dict[str, JSONFeatureValue],
    ) -> float:
        candidate_frame = pd.DataFrame([candidate], columns=self.feature_columns)
        candidate_frame = self._as_model_input_df(candidate_frame)
        return float(self.artifacts.model.predict_proba(candidate_frame)[0][0])

    @staticmethod
    def _empty_candidate_result() -> NNCECandidateResult:
        return NNCECandidateResult(
            candidate_features=None,
            source_reference_index=None,
            distance=None,
            probability_low_risk=None,
            target_satisfied=False,
            changed_immutable_features=(),
            changed_outside_selected_mutable_features=(),
        )

    def _build_summary(
        self,
        scenario_results: list[NNCEActionabilityScenarioResult],
    ) -> dict[str, Any]:
        pure_results = [result.pure_nnce for result in scenario_results]
        adapted_results = [result.adapted_nnce for result in scenario_results]
        pure_immutable_counts = [len(result.changed_immutable_features) for result in pure_results]
        violating_pure_immutable_counts = [count for count in pure_immutable_counts if count > 0]
        immutable_feature_counts: Counter[str] = Counter()
        for result in pure_results:
            immutable_feature_counts.update(result.changed_immutable_features)

        return {
            "scenario_count": len(scenario_results),
            "pure_nnce": {
                "immutable_violation_rate": _rate(
                    pure_results,
                    lambda item: item.immutable_violation,
                ),
                "outside_selected_mutable_violation_rate": _rate(
                    pure_results,
                    lambda item: item.outside_selected_mutable_violation,
                ),
                "average_changed_immutable_feature_count_all_scenarios": (
                    float(np.mean(pure_immutable_counts)) if pure_immutable_counts else 0.0
                ),
                "average_changed_immutable_feature_count_when_violating": (
                    float(np.mean(violating_pure_immutable_counts))
                    if violating_pure_immutable_counts
                    else 0.0
                ),
                "changed_immutable_feature_frequency": [
                    {
                        "feature_name": feature_name,
                        "changed_count": count,
                        "changed_rate": count / len(scenario_results) if scenario_results else 0.0,
                    }
                    for feature_name, count in immutable_feature_counts.most_common()
                ],
            },
            "adapted_nnce": {
                "immutable_violation_rate": _rate(
                    adapted_results,
                    lambda item: item.immutable_violation,
                ),
                "outside_selected_mutable_violation_rate": _rate(
                    adapted_results,
                    lambda item: item.outside_selected_mutable_violation,
                ),
            },
        }

    def _scenario_payload(
        self,
        result: NNCEActionabilityScenarioResult,
    ) -> dict[str, Any]:
        return {
            "name": result.scenario_name,
            "request_id": result.request_id,
            "mutable_allowed": list(result.mutable_allowed),
            "pure_nnce": _candidate_payload(result.pure_nnce),
            "adapted_nnce": _candidate_payload(result.adapted_nnce),
        }

    def _canonical_baseline(
        self,
        request: CounterfactualRequest,
    ) -> dict[str, JSONFeatureValue]:
        canonical = self.registry.canonicalize_feature_map(request.instance.features)
        return {
            feature_name: self._coerce_feature_value(feature_name, canonical[feature_name])
            for feature_name in self.feature_columns
        }

    def _reference_features(self, reference_index: int) -> dict[str, JSONFeatureValue]:
        row = self.reference_frame.iloc[reference_index]
        return {
            feature_name: self._coerce_feature_value(feature_name, row[feature_name])
            for feature_name in self.feature_columns
        }

    def _coerce_feature_value(
        self,
        feature_name: str,
        value: object,
    ) -> JSONFeatureValue:
        coerced = self.registry.coerce_value(feature_name, value)
        if isinstance(coerced, np.generic):
            coerced = coerced.item()
        if not isinstance(coerced, (int, float, bool, str)):
            coerced = float(str(coerced))
        return coerced

    def _as_model_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
        typed = frame.copy()
        for column in self.feature_columns:
            numeric = pd.to_numeric(typed[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"Feature '{column}' contains non-numeric value(s).")
            feature = self.registry.get(column)
            if feature is not None and (feature.is_binary or feature.feature_type == "ordinal"):
                typed[column] = numeric.round().astype("int64")
            else:
                typed[column] = numeric.astype("float64")
        return typed


def heom_distance(
    *,
    baseline: dict[str, JSONFeatureValue],
    candidate: dict[str, JSONFeatureValue],
    feature_names: tuple[str, ...],
    registry: FeatureRegistry,
) -> float:
    squared_total = 0.0
    for feature_name in feature_names:
        feature = registry.get(feature_name)
        squared_total += (
            _heom_component(
                feature=feature,
                baseline_value=baseline[feature_name],
                candidate_value=candidate[feature_name],
            )
            ** 2
        )
    return float(np.sqrt(squared_total))


def load_json_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment JSON payload must be an object")
    return payload


def _heom_component(
    *,
    feature: FeatureDefinition | None,
    baseline_value: JSONFeatureValue,
    candidate_value: JSONFeatureValue,
) -> float:
    if feature is not None and feature.is_binary:
        return 0.0 if _equal(baseline_value, candidate_value) else 1.0

    baseline_float = float(baseline_value)
    candidate_float = float(candidate_value)
    if feature is not None and feature.global_min is not None and feature.global_max is not None:
        span = max(float(feature.global_max) - float(feature.global_min), 1e-12)
    else:
        span = max(abs(baseline_float), abs(candidate_float), 1.0)
    return abs(candidate_float - baseline_float) / span


def _candidate_payload(result: NNCECandidateResult) -> dict[str, Any]:
    return {
        "source_reference_index": result.source_reference_index,
        "distance": result.distance,
        "probability_low_risk": result.probability_low_risk,
        "target_satisfied": result.target_satisfied,
        "immutable_violation": result.immutable_violation,
        "outside_selected_mutable_violation": result.outside_selected_mutable_violation,
        "changed_immutable_features": list(result.changed_immutable_features),
        "changed_outside_selected_mutable_features": list(
            result.changed_outside_selected_mutable_features
        ),
        "candidate_features": result.candidate_features,
    }


def _rate(
    items: list[NNCECandidateResult],
    predicate: Any,
) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def _changed(a: JSONFeatureValue, b: JSONFeatureValue) -> bool:
    return not _equal(a, b)


def _equal(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def successful_actionability_scenarios(
    scenarios: list[VerificationScenario],
) -> list[VerificationScenario]:
    return [
        scenario
        for scenario in scenarios
        if scenario.expectation.expected_status == Status.FEASIBLE
    ]
