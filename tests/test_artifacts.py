from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from diabetify_cf.engine.artifacts import _sha256_file, load_artifacts


def _write_artifacts(tmp_path: Path) -> dict[str, Path]:
    model_path = tmp_path / "model.pkl"
    columns_path = tmp_path / "columns.pkl"
    reference_path = tmp_path / "reference.csv"
    registry_path = tmp_path / "feature_registry.json"

    model_path.write_bytes(pickle.dumps(object()))
    columns_path.write_bytes(pickle.dumps(["BMI"]))
    pd.DataFrame({"BMI": [31.2, 28.0]}).to_csv(reference_path, index=False)
    registry_path.write_text(
        json.dumps(
            {
                "version": "test_v1",
                "features": [
                    {
                        "name": "BMI",
                        "type": "continuous",
                        "immutable": False,
                        "actionable": True,
                        "default_mutable": True,
                        "global_min": 10,
                        "global_max": 60,
                        "cost_weight": 1.0,
                        "preferred_direction": "decrease",
                        "aliases": ["bmi"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "columns": columns_path,
        "reference_data": reference_path,
        "feature_registry": registry_path,
    }


def test_load_artifacts_records_artifact_checksums(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)

    artifacts = load_artifacts(
        model_path=str(paths["model"]),
        columns_path=str(paths["columns"]),
        reference_data_path=str(paths["reference_data"]),
        feature_registry_path=str(paths["feature_registry"]),
    )

    assert artifacts.metadata is not None
    assert artifacts.metadata.checksums["model"] == _sha256_file(str(paths["model"]))
    assert artifacts.metadata.checksums["columns"] == _sha256_file(str(paths["columns"]))


def test_load_artifacts_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": {"model": {"sha256": "not-the-real-checksum"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_artifacts(
            model_path=str(paths["model"]),
            columns_path=str(paths["columns"]),
            reference_data_path=str(paths["reference_data"]),
            feature_registry_path=str(paths["feature_registry"]),
            artifact_manifest_path=str(manifest_path),
        )
