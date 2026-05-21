"""Feature metadata utilities used by the counterfactual engine.

The ML model only knows column names and numeric values. The engine needs a
layer above that to answer questions such as:
- which features are immutable,
- which values are valid,
- which aliases from incoming payloads map to model columns,
- which direction of change is clinically preferred.

This module turns `feature_registry.json` into a typed helper object that the
engine can query during validation and candidate ranking.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

FeatureValue = TypeVar("FeatureValue")


@dataclass(frozen=True)
class FeatureDefinition:
    """Runtime representation of one feature policy entry."""

    name: str
    feature_type: str
    immutable: bool
    actionable: bool
    default_mutable: bool
    global_min: float | None
    global_max: float | None
    cost_weight: float
    preferred_direction: str
    aliases: list[str]
    allowed_transitions: dict[int, tuple[int, ...]] | None = None

    @property
    def is_continuous(self) -> bool:
        return self.feature_type in {"continuous", "ordinal"}

    @property
    def is_binary(self) -> bool:
        return self.feature_type == "binary"


class FeatureRegistry:
    """In-memory lookup helper built from `feature_registry.json`."""

    def __init__(self, version: str, features: list[FeatureDefinition]) -> None:
        self.version = version
        self.features = features

        self._by_name: dict[str, FeatureDefinition] = {
            feature.name: feature for feature in features
        }
        self._alias_to_name: dict[str, str] = {}
        for feature in features:
            for alias in feature.aliases:
                self._alias_to_name[alias] = feature.name

    @classmethod
    def from_file(cls, path: str) -> FeatureRegistry:
        """Build a registry from a JSON feature registry file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = str(payload.get("version", "med_rule_v1"))
        features_raw = payload.get("features", [])
        features = [
            FeatureDefinition(
                name=item["name"],
                feature_type=item["type"],
                immutable=bool(item.get("immutable", False)),
                actionable=bool(item.get("actionable", True)),
                default_mutable=bool(item.get("default_mutable", True)),
                global_min=item.get("global_min"),
                global_max=item.get("global_max"),
                cost_weight=float(item.get("cost_weight", 1.0)),
                preferred_direction=cls._normalize_direction(
                    str(item.get("preferred_direction", "any")).lower()
                ),
                aliases=list(item.get("aliases", [])),
                allowed_transitions=cls._parse_allowed_transitions(item.get("allowed_transitions")),
            )
            for item in features_raw
        ]
        return cls(version=version, features=features)

    @classmethod
    def from_columns(cls, columns: list[str]) -> FeatureRegistry:
        """Create a permissive fallback registry from raw model columns."""
        features: list[FeatureDefinition] = []
        for name in columns:
            features.append(
                FeatureDefinition(
                    name=name,
                    feature_type="continuous",
                    immutable=False,
                    actionable=True,
                    default_mutable=True,
                    global_min=None,
                    global_max=None,
                    cost_weight=1.0,
                    preferred_direction="any",
                    aliases=[],
                    allowed_transitions=None,
                )
            )
        return cls(version="auto_v1", features=features)

    def resolve_name(self, name: str) -> str:
        """Resolve an alias to the canonical model feature name."""
        if name in self._by_name:
            return name
        return self._alias_to_name.get(name, name)

    def canonicalize_feature_map(self, raw: Mapping[str, FeatureValue]) -> dict[str, FeatureValue]:
        """Normalize request payload keys into canonical feature names."""
        canonical: dict[str, FeatureValue] = {}
        for key, value in raw.items():
            canonical[self.resolve_name(key)] = value
        return canonical

    def canonicalize_feature_names(self, names: list[str]) -> list[str]:
        """Normalize a list of feature names or aliases."""
        return [self.resolve_name(name) for name in names]

    def get(self, name: str) -> FeatureDefinition | None:
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def default_permitted_range(self, feature_names: list[str]) -> dict[str, list[float]]:
        """Return default search bounds for features with known min/max."""
        out: dict[str, list[float]] = {}
        for name in feature_names:
            feature = self.get(name)
            if feature is None:
                continue
            if feature.global_min is None or feature.global_max is None:
                continue
            out[name] = [feature.global_min, feature.global_max]
        return out

    def immutable_defaults(self) -> list[str]:
        """Return features that are globally immutable by policy."""
        return [feature.name for feature in self.features if feature.immutable]

    def continuous_features(self, feature_names: list[str]) -> list[str]:
        """Return the subset treated as continuous or ordinal by the engine."""
        continuous = []
        for name in feature_names:
            feature = self.get(name)
            if feature is not None and feature.is_continuous:
                continuous.append(name)
        return continuous

    def coerce_value(self, name: str, value: object) -> object:
        """Cast one value to the type expected for the given feature.

        This keeps request payloads, generated candidates, and model inputs in a
        consistent numeric form.
        """
        feature = self.get(name)
        if feature is None:
            return value
        if feature.is_binary:
            return int(round(float(cast(Any, value))))
        if feature.feature_type == "ordinal":
            return int(round(float(cast(Any, value))))
        if feature.feature_type == "continuous":
            return float(cast(Any, value))
        return value

    @staticmethod
    def _parse_allowed_transitions(
        raw: object,
    ) -> dict[int, tuple[int, ...]] | None:
        if not isinstance(raw, Mapping):
            return None

        parsed: dict[int, tuple[int, ...]] = {}
        for baseline_raw, targets_raw in raw.items():
            try:
                baseline = int(baseline_raw)
            except (TypeError, ValueError):
                continue

            if not isinstance(targets_raw, list):
                continue

            targets: list[int] = []
            for target_raw in targets_raw:
                try:
                    targets.append(int(target_raw))
                except (TypeError, ValueError):
                    continue

            parsed[baseline] = tuple(targets)

        return parsed or None

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        """Keep only supported direction values, fallback to `any`."""
        if direction in {"increase", "decrease", "any"}:
            return direction
        return "any"
