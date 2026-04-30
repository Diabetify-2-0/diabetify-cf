from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_path()

from diabetify_cf.config import Settings  # noqa: E402
from diabetify_cf.schemas import CounterfactualRequest  # noqa: E402
from experiments.engines.factory import build_experiment_engine  # noqa: E402
from experiments.scripts.run_benchmark import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    build_request_payload,
    load_config,
    load_effective_config,
    select_evaluation_rows,
)
from experiments.scripts.run_metadata import build_run_metadata  # noqa: E402


def _make_output_dir(output_root: Path, engine_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_{engine_name}_stability"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 1.0
    return len(a.intersection(b)) / len(union)


def _feature_range(feature_name: str, registry: Any) -> float:
    feature = registry.get(feature_name)
    if feature is None or feature.global_min is None or feature.global_max is None:
        return 1.0
    return max(float(feature.global_max) - float(feature.global_min), 1e-6)


def _delta_std_norm(deltas: list[dict[str, float]], registry: Any) -> float:
    features = sorted({feature for delta in deltas for feature in delta})
    if not features or len(deltas) < 2:
        return 0.0

    std_values: list[float] = []
    for feature_name in features:
        values = [float(delta.get(feature_name, 0.0)) for delta in deltas]
        std_values.append(statistics.pstdev(values) / _feature_range(feature_name, registry))
    return sum(std_values) / len(std_values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _aggregate_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(summary_rows),
        "mean_feasible_rate": _mean(summary_rows, "feasible_rate"),
        "mean_jaccard_changed_features": _mean(summary_rows, "mean_jaccard_changed_features"),
        "mean_stability_std_norm": _mean(summary_rows, "stability_std_norm"),
    }


def evaluate_stability(
    config: dict[str, Any],
    output_root: Path,
    repeat_count: int,
    config_path: Path | None = None,
) -> Path:
    settings = Settings()
    engine_name = str(config.get("engine", "dice")).lower()
    adapter = build_experiment_engine(engine_name, settings=settings)
    if adapter.engine.artifacts is None:
        raise RuntimeError(
            adapter.engine.initialization_error or f"{engine_name} artifacts are not ready."
        )

    artifacts = adapter.engine.artifacts
    output_dir = _make_output_dir(output_root, engine_name)
    selected = select_evaluation_rows(
        reference_data=artifacts.reference_data,
        model=artifacts.model,
        feature_columns=artifacts.feature_columns,
        limit=int(config.get("limit", 10)),
        only_high_risk=bool(config.get("only_high_risk", True)),
        random_seed=int(config.get("random_seed", 42)),
    )

    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case_index, (_, row) in enumerate(selected.iterrows(), start=1):
        deltas: list[dict[str, float]] = []
        changed_sets: list[set[str]] = []
        feasible_count = 0

        for repeat_index in range(repeat_count):
            run_config = dict(config)
            run_config["random_seed"] = int(config.get("random_seed", 42)) + repeat_index
            payload = build_request_payload(
                row=row.to_dict(),
                index=case_index,
                feature_columns=artifacts.feature_columns,
                registry=artifacts.feature_registry,
                config=run_config,
            )
            request = CounterfactualRequest.model_validate(payload)
            result = adapter.generate(request)
            top_candidate = result.candidates[0] if result.candidates else {}
            delta = top_candidate.get("delta", {}) if isinstance(top_candidate, dict) else {}
            if not isinstance(delta, dict):
                delta = {}
            delta_float = {str(key): float(value) for key, value in delta.items()}
            changed = set(delta_float)
            deltas.append(delta_float)
            changed_sets.append(changed)
            if result.status == "FEASIBLE":
                feasible_count += 1

            run_rows.append(
                {
                    "case_id": f"case-{case_index:05d}",
                    "repeat_index": repeat_index,
                    "request_id": result.request_id,
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "runtime_ms": result.runtime_ms,
                    "candidate_count": result.candidate_count,
                    "top_delta": json.dumps(delta_float, ensure_ascii=True),
                }
            )

        reference_changed = changed_sets[0] if changed_sets else set()
        jaccards = [_jaccard(reference_changed, item) for item in changed_sets[1:]]
        summary_rows.append(
            {
                "case_id": f"case-{case_index:05d}",
                "repeat_count": repeat_count,
                "feasible_count": feasible_count,
                "feasible_rate": feasible_count / repeat_count if repeat_count else 0.0,
                "mean_jaccard_changed_features": sum(jaccards) / len(jaccards) if jaccards else 1.0,
                "stability_std_norm": _delta_std_norm(deltas, artifacts.feature_registry),
            }
        )

    run_config = {
        "config": config,
        "repeat_count": repeat_count,
        "metadata": build_run_metadata(
            repo_root=REPO_ROOT,
            run_type="stability",
            engine_name=engine_name,
            config_path=config_path,
        ),
        "settings": {
            "model_path": settings.model_path,
            "columns_path": settings.columns_path,
            "reference_data_path": settings.reference_data_path,
            "feature_registry_path": settings.feature_registry_path,
        },
        "selected_case_count": len(selected),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "stability_runs.csv", run_rows)
    _write_csv(output_dir / "stability_summary.csv", summary_rows)
    _write_csv(output_dir / "stability_aggregate.csv", [_aggregate_summary(summary_rows)])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repeated-run stability.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Merged effective config. Prefer --engine-config and --scenario-config.",
    )
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG_PATH)
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=Path("experiments/configs/scenarios/stability.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeat-count", type=int, default=10)
    args = parser.parse_args()

    config_path = args.config or args.scenario_config
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_effective_config(args.engine_config, args.scenario_config)

    output_dir = evaluate_stability(
        config=config,
        output_root=args.output_root,
        repeat_count=args.repeat_count,
        config_path=config_path,
    )
    print(f"Stability output written to {output_dir}")


if __name__ == "__main__":
    main()
