from __future__ import annotations

import argparse
import json
from pathlib import Path

from diabetify_cf.config import SERVICE_ROOT, Settings
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblationConfig,
    artifact_sha256,
)
from diabetify_cf.experiments.nn_projection_gate_audit import NNProjectionGateAudit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sequential gate audit over NN projection candidates."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional projection configuration JSON. Uses built-in defaults when omitted.",
    )
    parser.add_argument(
        "--profile-input",
        default=str(Path("evaluation") / "fixtures" / "profile_input.json"),
        help="Scenario fixture JSON to audit.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path("evaluation") / "reports" / "ablation" / "nn_projection" / "gate_audit.json"
        ),
        help="Path for the sequential gate audit report.",
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
    experiment = NNProjectionGateAudit(
        engine=engine,
        config=config,
        profiles_path=args.profile_input,
    )
    payload = experiment.run()
    payload["artifacts"] = {
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = payload["summary"]
    print(f"Profiles audited: {len(payload['profiles'])}")
    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Failed directional: {summary['failed_directional']}")
    print(f"Failed transition: {summary['failed_transition']}")
    print(f"Failed medical: {summary['failed_medical']}")
    print(f"Failed target: {summary['failed_target']}")
    print(f"Valid remaining: {summary['valid_remaining']}")
    print(f"Report written to: {output_path}")


def _portable_artifact_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(SERVICE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
