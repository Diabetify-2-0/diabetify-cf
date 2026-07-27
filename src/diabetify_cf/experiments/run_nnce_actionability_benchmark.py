from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diabetify_cf.config import SERVICE_ROOT, Settings
from diabetify_cf.engine.artifacts import load_artifacts
from diabetify_cf.experiments.nn_projection_ablation import artifact_sha256
from diabetify_cf.experiments.nnce_actionability_benchmark import (
    NNCEActionabilityBenchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pure NNCE vs actionability-adapted NNCE benchmark on "
            "actionability fixtures."
        )
    )
    parser.add_argument(
        "--scenarios",
        default=str(Path("evaluation") / "fixtures" / "actionability_profiles.json"),
        help="Path to actionability scenario JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path("evaluation") / "reports" / "experiments" / "nnce_actionability_benchmark.json"
        ),
        help="Path for the benchmark report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    artifacts = load_artifacts(
        model_path=settings.model_path,
        columns_path=settings.columns_path,
        reference_data_path=settings.reference_data_path,
        feature_registry_path=settings.feature_registry_path,
    )
    benchmark = NNCEActionabilityBenchmark.from_fixture_path(
        artifacts=artifacts,
        scenarios_path=args.scenarios,
    )
    payload = benchmark.run()
    payload["artifacts"] = {
        "model_path": _portable_artifact_path(settings.model_path),
        "model_sha256": artifact_sha256(settings.model_path),
        "columns_path": _portable_artifact_path(settings.columns_path),
        "columns_sha256": artifact_sha256(settings.columns_path),
        "reference_data_path": _portable_artifact_path(settings.reference_data_path),
        "reference_data_sha256": artifact_sha256(settings.reference_data_path),
        "feature_registry_path": _portable_artifact_path(settings.feature_registry_path),
        "feature_registry_sha256": artifact_sha256(settings.feature_registry_path),
    }

    output_path = Path(args.output)
    _write_json(output_path, payload)
    summary = payload["summary"]
    print(f"Scenarios evaluated: {summary['scenario_count']}")
    print(
        "Immutable violation rate (pure -> adapted): "
        f"{summary['pure_nnce']['immutable_violation_rate']:.3f} -> "
        f"{summary['adapted_nnce']['immutable_violation_rate']:.3f}"
    )
    print(
        "Outside-selected mutable violation rate (pure -> adapted): "
        f"{summary['pure_nnce']['outside_selected_mutable_violation_rate']:.3f} -> "
        f"{summary['adapted_nnce']['outside_selected_mutable_violation_rate']:.3f}"
    )
    print(f"Report written to: {output_path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_artifact_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(SERVICE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
