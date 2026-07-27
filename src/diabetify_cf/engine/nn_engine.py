from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from diabetify_cf.config import Settings
from diabetify_cf.engine.feature_registry import FeatureDefinition
from diabetify_cf.engine.shared import ArtifactBackedCounterfactualEngine, PreparedRequest
from diabetify_cf.schemas import CounterfactualRequest


@dataclass(frozen=True)
class NearestNeighborOptions:
    candidate_pool_size: int = 256
    max_neighbors: int = 64

    @classmethod
    def from_settings(cls, settings: Settings) -> NearestNeighborOptions:
        return cls(
            candidate_pool_size=settings.nn_candidate_pool_size,
            max_neighbors=settings.nn_max_neighbors,
        )


class NearestNeighborCounterfactualEngine(ArtifactBackedCounterfactualEngine):
    engine_version = "nn_engine_v1"

    def __init__(
        self,
        *,
        model_path: str = "",
        columns_path: str = "",
        reference_data_path: str = "",
        feature_registry_path: str = "",
        options: NearestNeighborOptions | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            columns_path=columns_path,
            reference_data_path=reference_data_path,
            feature_registry_path=feature_registry_path,
        )
        self.options = options or NearestNeighborOptions()

    def _generate_raw_candidates(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
    ) -> pd.DataFrame:
        assert self.artifacts is not None

        baseline = prepared.instance_features
        mutable_allowed = prepared.mutable_allowed
        if not mutable_allowed:
            return pd.DataFrame()

        reference_frame = self.artifacts.reference_data[self.artifacts.feature_columns].copy()
        reference_frame = self._as_model_input_df(reference_frame)
        low_risk_probabilities = self.artifacts.model.predict_proba(reference_frame)[:, 0]
        eligible_mask = low_risk_probabilities >= float(request.target.min_target_probability)
        eligible = reference_frame.loc[eligible_mask].copy()
        eligible_probabilities = low_risk_probabilities[eligible_mask]

        if eligible.empty:
            top_indices = np.argsort(low_risk_probabilities)[::-1][
                : self.options.candidate_pool_size
            ]
            eligible = reference_frame.iloc[top_indices].copy()
            eligible_probabilities = low_risk_probabilities[top_indices]

        ranked_neighbors = self._rank_neighbors(
            eligible=eligible,
            eligible_probabilities=eligible_probabilities,
            baseline=baseline,
            mutable_allowed=mutable_allowed,
            prepared=prepared,
        )
        ranked_neighbors = ranked_neighbors[: self.options.candidate_pool_size]
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
        return self._as_model_input_df(ordered)

    def _rank_neighbors(
        self,
        *,
        eligible: pd.DataFrame,
        eligible_probabilities: np.ndarray,
        baseline: dict[str, Any],
        mutable_allowed: list[str],
        prepared: PreparedRequest,
    ) -> list[pd.Series]:
        ranked: list[tuple[tuple[float, float, int, float], pd.Series]] = []
        for row, low_risk_probability in zip(
            eligible.itertuples(index=False, name=None),
            eligible_probabilities,
            strict=True,
        ):
            series = pd.Series(row, index=eligible.columns)
            candidate_features = self._series_to_feature_map(series=series, prepared=prepared)
            actionable_candidate_features = {
                feature_name: candidate_features[feature_name]
                for feature_name in mutable_allowed
                if feature_name in candidate_features
            }
            actionable_baseline = {
                feature_name: baseline[feature_name]
                for feature_name in mutable_allowed
                if feature_name in baseline
            }
            heom_distance = self._normalized_l1(
                actionable_candidate_features,
                actionable_baseline,
                prepared.registry,
            )
            action_cost = 0.0
            changed_count = 0
            for feature_name in mutable_allowed:
                delta = abs(float(series[feature_name]) - float(baseline[feature_name]))
                if delta < 1e-9:
                    continue
                changed_count += 1
                action_cost += self._weighted_normalized_delta(
                    feature_name=feature_name,
                    delta=delta,
                    prepared=prepared,
                )
            score = (
                heom_distance,
                action_cost,
                changed_count,
                -float(low_risk_probability),
            )
            ranked.append((score, series))
        ranked.sort(key=lambda item: item[0])
        return [series for _, series in ranked]

    def _series_to_feature_map(
        self,
        *,
        series: pd.Series,
        prepared: PreparedRequest,
    ) -> dict[str, Any]:
        return {
            feature_name: prepared.registry.coerce_value(feature_name, series[feature_name])
            for feature_name in prepared.model_columns
        }

    def _project_neighbors(
        self,
        *,
        ranked_neighbors: list[pd.Series],
        baseline: dict[str, Any],
        mutable_allowed: list[str],
        prepared: PreparedRequest,
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        effective_max_changed_features = len(mutable_allowed)
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
            self._append_projection(projection=full_projection, projected=projected, seen=seen)

            sparse_limit = min(effective_max_changed_features, len(changed_features))
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
        prepared: PreparedRequest,
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
        prepared: PreparedRequest,
    ) -> float:
        feature = prepared.registry.get(feature_name)
        cost_weight = 1.0 if feature is None else float(feature.cost_weight)
        span = self._feature_span(feature_name=feature_name, feature=feature)
        return (delta / span) * cost_weight

    def _feature_span(
        self,
        *,
        feature_name: str,
        feature: FeatureDefinition | None,
    ) -> float:
        if (
            feature is not None
            and feature.global_min is not None
            and feature.global_max is not None
        ):
            return max(float(feature.global_max) - float(feature.global_min), 1e-6)
        assert self.artifacts is not None
        series = pd.to_numeric(
            self.artifacts.reference_data[feature_name], errors="coerce"
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
