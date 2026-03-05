from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    feature_type: str
    immutable: bool
    actionable: bool
    default_mutable: bool
    global_min: Optional[float]
    global_max: Optional[float]
    cost_weight: float
    preferred_direction: str
    aliases: List[str]

    @property
    def is_continuous(self) -> bool:
        return self.feature_type in {"continuous", "ordinal"}

    @property
    def is_binary(self) -> bool:
        return self.feature_type == "binary"


class FeatureRegistry:
    def __init__(self, version: str, features: List[FeatureDefinition]) -> None:
        self.version = version
        self.features = features

        self._by_name: Dict[str, FeatureDefinition] = {
            feature.name: feature for feature in features
        }
        self._alias_to_name: Dict[str, str] = {}
        for feature in features:
            for alias in feature.aliases:
                self._alias_to_name[alias] = feature.name

    @classmethod
    def from_file(cls, path: str) -> "FeatureRegistry":
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
            )
            for item in features_raw
        ]
        return cls(version=version, features=features)

    @classmethod
    def from_columns(cls, columns: List[str]) -> "FeatureRegistry":
        features: List[FeatureDefinition] = []
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
                )
            )
        return cls(version="auto_v1", features=features)

    def resolve_name(self, name: str) -> str:
        if name in self._by_name:
            return name
        return self._alias_to_name.get(name, name)

    def canonicalize_feature_map(self, raw: Dict[str, object]) -> Dict[str, object]:
        canonical: Dict[str, object] = {}
        for key, value in raw.items():
            canonical[self.resolve_name(key)] = value
        return canonical

    def canonicalize_feature_names(self, names: List[str]) -> List[str]:
        return [self.resolve_name(name) for name in names]

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def default_permitted_range(self, feature_names: List[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for name in feature_names:
            feature = self.get(name)
            if feature is None:
                continue
            if feature.global_min is None or feature.global_max is None:
                continue
            out[name] = [feature.global_min, feature.global_max]
        return out

    def immutable_defaults(self) -> List[str]:
        return [feature.name for feature in self.features if feature.immutable]

    def continuous_features(self, feature_names: List[str]) -> List[str]:
        return [name for name in feature_names if (self.get(name) and self.get(name).is_continuous)]

    def coerce_value(self, name: str, value: object) -> object:
        feature = self.get(name)
        if feature is None:
            return value
        if feature.is_binary:
            return int(round(float(value)))
        if feature.feature_type == "ordinal":
            return int(round(float(value)))
        if feature.feature_type == "continuous":
            return float(value)
        return value

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        if direction in {"increase", "decrease", "any"}:
            return direction
        return "any"
