from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from diabetify_cf.engine.feature_registry import FeatureRegistry


@dataclass
class ModelArtifacts:
    model: Any
    feature_columns: list[str]
    reference_data: pd.DataFrame
    feature_registry: FeatureRegistry
    lof_model: LocalOutlierFactor | None


def load_artifacts(
    model_path: str,
    columns_path: str,
    reference_data_path: str,
    feature_registry_path: str,
) -> ModelArtifacts:
    if not model_path or not columns_path:
        raise ValueError("model path and columns path are required for real engine mode")

    model_file = Path(model_path)
    columns_file = Path(columns_path)

    if not model_file.exists():
        raise FileNotFoundError(f"model file not found: {model_file}")
    if not columns_file.exists():
        raise FileNotFoundError(f"columns file not found: {columns_file}")

    with model_file.open("rb") as f:
        model = pickle.load(f)
    with columns_file.open("rb") as f:
        feature_columns = list(pickle.load(f))

    feature_registry = _load_feature_registry(feature_registry_path, feature_columns)
    reference_data = _load_reference_data(reference_data_path, feature_columns)
    lof_model = _build_lof_model(reference_data)

    return ModelArtifacts(
        model=model,
        feature_columns=feature_columns,
        reference_data=reference_data,
        feature_registry=feature_registry,
        lof_model=lof_model,
    )


def _load_feature_registry(path: str, feature_columns: list[str]) -> FeatureRegistry:
    if not path:
        return FeatureRegistry.from_columns(feature_columns)

    file = Path(path)
    if not file.exists():
        return FeatureRegistry.from_columns(feature_columns)

    registry = FeatureRegistry.from_file(str(file))
    # Keep only model columns in model order, fallback unknown to auto definition.
    definitions = []
    for column in feature_columns:
        feature = registry.get(column)
        if feature is None:
            auto = FeatureRegistry.from_columns([column]).get(column)
            if auto is not None:
                definitions.append(auto)
            continue
        definitions.append(feature)
    return FeatureRegistry(version=registry.version, features=definitions)


def _load_reference_data(path: str, feature_columns: list[str]) -> pd.DataFrame:
    if not path:
        return _empty_reference(feature_columns)

    file = Path(path)
    if not file.exists():
        return _empty_reference(feature_columns)

    suffix = file.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            data = pd.read_parquet(file)
        except Exception:
            data = _empty_reference(feature_columns)
    elif suffix == ".csv":
        data = pd.read_csv(file)
    elif suffix == ".json":
        data = pd.read_json(file)
    else:
        data = _empty_reference(feature_columns)

    for column in feature_columns:
        if column not in data.columns:
            data[column] = 0.0
    return data[feature_columns].copy()


def _empty_reference(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)


def _build_lof_model(reference_data: pd.DataFrame) -> LocalOutlierFactor | None:
    if len(reference_data) < 2:
        return None

    # novelty=True allows scoring unseen candidates.
    lof = LocalOutlierFactor(n_neighbors=min(20, len(reference_data) - 1), novelty=True)
    lof.fit(reference_data)
    return lof
