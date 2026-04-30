from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_path()

from diabetify_cf.config import Settings  # noqa: E402
from diabetify_cf.engine.feature_registry import FeatureRegistry  # noqa: E402
from diabetify_cf.schemas import CounterfactualRequest  # noqa: E402
from experiments.engines.base import EngineRunResult  # noqa: E402
from experiments.engines.factory import build_experiment_engine  # noqa: E402
from experiments.evaluation import evaluate_candidate  # noqa: E402
from experiments.scripts.run_metadata import build_run_metadata  # noqa: E402

DEFAULT_ENGINE_CONFIG_PATH = Path("experiments/configs/engines/dice.json")
DEFAULT_SCENARIO_CONFIG_PATH = Path("experiments/configs/scenarios/all_mutable.json")
DEFAULT_OUTPUT_ROOT = Path("experiments/results")
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_configs(engine_config: dict[str, Any], scenario_config: dict[str, Any]) -> dict[str, Any]:
    return {**scenario_config, **engine_config}


def load_effective_config(engine_config_path: Path, scenario_config_path: Path) -> dict[str, Any]:
    return merge_configs(
        engine_config=load_config(engine_config_path),
        scenario_config=load_config(scenario_config_path),
    )


def _default_mutable_allowed(registry: FeatureRegistry, feature_columns: list[str]) -> list[str]:
    out: list[str] = []
    for feature_name in feature_columns:
        feature = registry.get(feature_name)
        if feature is None:
            continue
        if feature.default_mutable and feature.actionable and not feature.immutable:
            out.append(feature_name)
    return out


def _default_immutable_features(registry: FeatureRegistry) -> list[str]:
    return registry.immutable_defaults()


def _default_feature_bounds(
    registry: FeatureRegistry, mutable_allowed: list[str]
) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for feature_name in mutable_allowed:
        feature = registry.get(feature_name)
        if feature is None:
            continue
        if feature.global_min is None or feature.global_max is None:
            continue
        bounds[feature_name] = {
            "min": float(feature.global_min),
            "max": float(feature.global_max),
        }
    return bounds


def build_request_payload(
    *,
    row: dict[str, Any],
    index: int,
    feature_columns: list[str],
    registry: FeatureRegistry,
    config: dict[str, Any],
) -> dict[str, Any]:
    mutable_allowed = list(config.get("mutable_allowed") or [])
    if not mutable_allowed and bool(config.get("use_default_mutable", True)):
        mutable_allowed = _default_mutable_allowed(registry, feature_columns)

    immutable_features = list(config.get("immutable_features") or [])
    if not immutable_features and bool(config.get("use_default_immutable", True)):
        immutable_features = _default_immutable_features(registry)

    feature_bounds = dict(config.get("feature_bounds") or {})
    if not feature_bounds and bool(config.get("use_default_feature_bounds", True)):
        feature_bounds = _default_feature_bounds(registry, mutable_allowed)

    return {
        "request_id": f"exp-{index:05d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": str(config.get("model_version", "xgb_experiment")),
        "target": {
            "target_class": str(config.get("target_class", "low_risk")),
            "min_target_probability": float(config.get("min_target_probability", 0.5)),
        },
        "instance": {"features": {name: row[name] for name in feature_columns}},
        "constraints": {
            "immutable_features": immutable_features,
            "mutable_allowed": mutable_allowed,
            "feature_bounds": feature_bounds,
            "must_not_change": list(config.get("must_not_change") or []),
            "medical_rule_set_version": str(config.get("medical_rule_set_version", "med_rule_v1")),
        },
        "generation": {
            "total_cfs": int(config.get("total_cfs", 3)),
            "method": str(config.get("generation_method", "dice_genetic")),
            "random_seed": int(config.get("random_seed", 42)) + index,
            "timeout_ms": int(config.get("timeout_ms", 5000)),
        },
        "preferences": {
            "cost_weights": dict(config.get("cost_weights") or {}),
            "objective_weights": dict(config.get("objective_weights") or {}),
        },
    }


def _prediction_high_risk_mask(model: Any, frame: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        high_risk = probabilities[:, 1]
    else:
        high_risk = model.predict(frame)
    return pd.Series(high_risk >= 0.5, index=frame.index)


def select_evaluation_rows(
    *,
    reference_data: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    limit: int,
    only_high_risk: bool,
    random_seed: int,
) -> pd.DataFrame:
    candidates = reference_data[feature_columns].copy()
    if only_high_risk:
        mask = _prediction_high_risk_mask(model, candidates)
        candidates = candidates.loc[mask]

    if candidates.empty:
        return candidates

    sample_size = min(max(limit, 1), len(candidates))
    return candidates.sample(n=sample_size, random_state=random_seed)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _candidate_rows(
    *,
    result: EngineRunResult,
    request: CounterfactualRequest,
    baseline: dict[str, Any],
    registry: FeatureRegistry,
    max_lof_score: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in result.candidates:
        metrics = candidate.get("metrics", {})
        prediction = candidate.get("prediction", {})
        evaluation = evaluate_candidate(
            candidate=candidate,
            baseline=baseline,
            request=request,
            registry=registry,
            max_lof_score=max_lof_score,
        )
        rows.append(
            {
                "engine_name": result.engine_name,
                "request_id": result.request_id,
                "candidate_id": candidate.get("candidate_id"),
                "status": result.status,
                "probability_low_risk": prediction.get("probability_low_risk"),
                "distance_l1": metrics.get("distance_l1"),
                "changed_feature_count": metrics.get("changed_feature_count"),
                "lof_score": metrics.get("lof_score"),
                "delta": json.dumps(candidate.get("delta", {}), ensure_ascii=True),
                **evaluation,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_output_dir(output_root: Path, engine_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_{engine_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_benchmark(
    config: dict[str, Any],
    output_root: Path,
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
    limit = int(config.get("limit", 10))
    random_seed = int(config.get("random_seed", 42))
    output_dir = _make_output_dir(output_root, engine_name)

    selected = select_evaluation_rows(
        reference_data=artifacts.reference_data,
        model=artifacts.model,
        feature_columns=artifacts.feature_columns,
        limit=limit,
        only_high_risk=bool(config.get("only_high_risk", True)),
        random_seed=random_seed,
    )

    case_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for case_index, (_, row) in enumerate(selected.iterrows(), start=1):
        payload = build_request_payload(
            row=row.to_dict(),
            index=case_index,
            feature_columns=artifacts.feature_columns,
            registry=artifacts.feature_registry,
            config=config,
        )
        request = CounterfactualRequest.model_validate(payload)
        result = adapter.generate(request)
        case_records.append(result.to_case_record())
        candidate_records.extend(
            _candidate_rows(
                result=result,
                request=request,
                baseline=row.to_dict(),
                registry=artifacts.feature_registry,
                max_lof_score=settings.max_lof_score,
            )
        )

    run_config = {
        "config": config,
        "metadata": build_run_metadata(
            repo_root=REPO_ROOT,
            run_type="benchmark",
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
    _write_jsonl(output_dir / "cases.jsonl", case_records)
    _write_csv(output_dir / "candidates.csv", candidate_records)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run counterfactual engine benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Merged effective config. Prefer --engine-config and --scenario-config.",
    )
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG_PATH)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    config_path = args.config or args.scenario_config
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_effective_config(args.engine_config, args.scenario_config)

    output_dir = run_benchmark(
        config=config,
        output_root=args.output_root,
        config_path=config_path,
    )
    print(f"Benchmark output written to {output_dir}")


if __name__ == "__main__":
    main()
