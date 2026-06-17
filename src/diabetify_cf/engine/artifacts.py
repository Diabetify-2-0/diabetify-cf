from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from diabetify_cf.engine.feature_registry import FeatureRegistry


@dataclass(frozen=True)
class ArtifactMetadata:
    version: str
    checksums: dict[str, str]


@dataclass
class ModelArtifacts:
    model: Any
    feature_columns: list[str]
    reference_data: pd.DataFrame
    feature_registry: FeatureRegistry
    lof_model: Pipeline | None
    metadata: ArtifactMetadata | None = None


def load_artifacts(
    model_path: str,
    columns_path: str,
    reference_data_path: str,
    feature_registry_path: str,
    artifact_manifest_path: str = "",
) -> ModelArtifacts:
    """Load all artifacts needed to run the real counterfactual engine."""

    if not model_path or not columns_path:
        raise ValueError("model path and columns path are required for real engine mode")

    model_file = Path(model_path)
    columns_file = Path(columns_path)

    if not model_file.exists():
        raise FileNotFoundError(f"model file not found: {model_file}")
    if not columns_file.exists():
        raise FileNotFoundError(f"columns file not found: {columns_file}")

    metadata = _build_artifact_metadata(
        model_path=model_path,
        columns_path=columns_path,
        reference_data_path=reference_data_path,
        feature_registry_path=feature_registry_path,
    )
    _validate_artifact_manifest(artifact_manifest_path, metadata)

    with model_file.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*If you are loading a serialized model.*",
                category=UserWarning,
            )
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
        metadata=metadata,
    )


def _build_artifact_metadata(
    *,
    model_path: str,
    columns_path: str,
    reference_data_path: str,
    feature_registry_path: str,
) -> ArtifactMetadata:
    paths = {
        "model": model_path,
        "columns": columns_path,
        "reference_data": reference_data_path,
        "feature_registry": feature_registry_path,
    }
    checksums = {
        name: _sha256_file(path) for name, path in paths.items() if path and Path(path).exists()
    }
    return ArtifactMetadata(version="artifact_manifest_v1", checksums=checksums)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_manifest(
    manifest_path: str,
    metadata: ArtifactMetadata,
) -> None:
    if not manifest_path:
        return

    file = Path(manifest_path)
    if not file.exists():
        return

    manifest = json.loads(file.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts", {})
    if not isinstance(expected, dict):
        raise ValueError("artifact manifest 'artifacts' must be an object")

    for artifact_name, expected_item in expected.items():
        if not isinstance(expected_item, dict):
            continue
        expected_sha = expected_item.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            continue
        actual_sha = metadata.checksums.get(str(artifact_name))
        if actual_sha != expected_sha:
            raise ValueError(f"artifact checksum mismatch for '{artifact_name}'")


def _load_feature_registry(path: str, feature_columns: list[str]) -> FeatureRegistry:
    """Load feature metadata and align it to the model column order."""

    if not path:
        return FeatureRegistry.from_columns(feature_columns)

    file = Path(path)
    if not file.exists():
        return FeatureRegistry.from_columns(feature_columns)

    registry = FeatureRegistry.from_file(str(file))

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
    """Load reference data used by the NN engine and LOF plausibility scoring."""

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
    """Return a minimal placeholder frame when real reference data is absent."""
    return pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)


def _build_lof_model(reference_data: pd.DataFrame) -> Pipeline | None:
    if len(reference_data) < 2:
        return None

    lof = make_pipeline(
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=min(20, len(reference_data) - 1), novelty=True),
    )
    lof.fit(reference_data.to_numpy())
    return lof
