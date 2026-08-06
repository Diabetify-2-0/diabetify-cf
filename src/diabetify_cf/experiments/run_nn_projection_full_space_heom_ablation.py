from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diabetify_cf.config import SERVICE_ROOT, Settings
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblationConfig,
    NNProjectionFullSpaceHeomAblation,
    artifact_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the target-gate-only NN full-vs-prefix-sparse ablation study using "
            "full-feature HEOM neighbor ranking."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional experiment configuration JSON. Uses built-in defaults when omitted.",
    )
    parser.add_argument(
        "--profile-input",
        default=str(Path("evaluation") / "fixtures" / "profile_input.json"),
        help=(
            "Scenario fixture JSON to use as fixed ablation profile input. Set to an "
            "empty string to use deterministic reference-data sampling instead."
        ),
    )
    parser.add_argument(
        "--report-output",
        default=str(
            Path("evaluation")
            / "reports"
            / "ablation"
            / "nn_projection"
            / "projection_ablation_full_space_heom_report.json"
        ),
        help="Path for the detailed comparison report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        NNProjectionAblationConfig.from_file(args.config)
        if args.config
        else NNProjectionAblationConfig.defaults()
    )
    settings = Settings()
    engine = NearestNeighborCounterfactualEngine(
        model_path=settings.model_path,
        columns_path=settings.columns_path,
        reference_data_path=settings.reference_data_path,
        feature_registry_path=settings.feature_registry_path,
        options=NearestNeighborOptions(
            candidate_pool_size=config.candidate_pool_size,
            max_neighbors=config.max_neighbors,
        ),
    )
    experiment = NNProjectionFullSpaceHeomAblation(
        engine=engine,
        config=config,
        profile_input_path=args.profile_input or None,
    )
    _profiles_payload, report_payload = experiment.run()

    artifact_metadata: dict[str, Any] = {
        "model_path": _portable_artifact_path(settings.model_path),
        "model_sha256": artifact_sha256(settings.model_path),
        "columns_path": _portable_artifact_path(settings.columns_path),
        "columns_sha256": artifact_sha256(settings.columns_path),
        "reference_data_path": _portable_artifact_path(settings.reference_data_path),
        "reference_data_sha256": artifact_sha256(settings.reference_data_path),
        "feature_registry_path": _portable_artifact_path(settings.feature_registry_path),
        "feature_registry_sha256": artifact_sha256(settings.feature_registry_path),
        "engine_version": engine.engine_version,
    }
    report_payload["artifacts"] = artifact_metadata

    report_path = Path(args.report_output)
    _write_json(report_path, report_payload)

    summary = report_payload["summary"]
    print(f"Selected valid profile pairs: {summary['valid_pair_count']}")
    print(
        "Mean proximity (full -> sparse): "
        f"{summary['full_mean_proximity']:.6f} -> "
        f"{summary['sparse_mean_proximity']:.6f}"
    )
    print(
        "Mean changed features (full -> sparse): "
        f"{summary['full_mean_changed_feature_count']:.3f} -> "
        f"{summary['sparse_mean_changed_feature_count']:.3f}"
    )
    print(f"Report written to: {report_path}")


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
