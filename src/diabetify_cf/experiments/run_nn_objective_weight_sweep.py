from __future__ import annotations

import argparse
import json
from pathlib import Path

from diabetify_cf.config import SERVICE_ROOT, Settings
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)
from diabetify_cf.experiments.nn_objective_selection_sensitivity import (
    OBJECTIVE_WEIGHT_SWEEP_CONFIGS,
    NNObjectiveSelectionSensitivity,
)
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblationConfig,
    artifact_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep objective weights between proximity and LOF plausibility for valid "
            "projection candidates."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional projection configuration JSON. Uses built-in defaults when omitted.",
    )
    parser.add_argument(
        "--gate-audit",
        default=str(
            Path("evaluation") / "reports" / "ablation" / "nn_projection" / "gate_audit.json"
        ),
        help="Gate audit JSON produced by run_nn_projection_gate_audit.",
    )
    parser.add_argument(
        "--scenario-input",
        action="append",
        default=[],
        help=(
            "Scenario fixture JSON to use as profile input. Can be provided multiple "
            "times. When set, --gate-audit is ignored."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path("evaluation")
            / "reports"
            / "ablation"
            / "nn_projection"
            / "objective_weight_sweep.json"
        ),
        help="Path for the objective weight sweep report.",
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
    experiment = NNObjectiveSelectionSensitivity(
        engine=engine,
        config=config,
        gate_audit_path=None if args.scenario_input else args.gate_audit,
        scenario_paths=tuple(args.scenario_input),
        objective_configs=OBJECTIVE_WEIGHT_SWEEP_CONFIGS,
        experiment_name="nn_objective_weight_sweep",
        description=(
            "Compare best valid projection candidates under proximity-only, plausibility-only, "
            "and mixed proximity-plausibility objective weights."
        ),
        include_diagnostics=True,
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

    print(f"Profiles evaluated: {payload['profile_count']}")
    for config_name, summary in payload["summary_by_objective"].items():
        print(
            f"{config_name}: "
            f"mean proximity={summary['mean_proximity']:.6f}, "
            f"mean LOF={summary['mean_plausibility_lof']:.6f}"
        )
    print(f"Report written to: {output_path}")


def _portable_artifact_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(SERVICE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
