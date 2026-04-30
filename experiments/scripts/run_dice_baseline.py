from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
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

from experiments.scripts.collect_results import collect_results  # noqa: E402
from experiments.scripts.run_benchmark import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    load_config,
    merge_configs,
)
from experiments.scripts.run_scenarios import DEFAULT_SCENARIO_CONFIGS  # noqa: E402
from experiments.scripts.summarize_results import summarize_run, write_summary_csv  # noqa: E402

DEFAULT_STABILITY_CONFIG = Path("experiments/configs/scenarios/stability.json")
DEFAULT_SCENARIO_TIMEOUT_SECONDS = 300
DEFAULT_STABILITY_TIMEOUT_SECONDS = 300


def _make_baseline_root(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    baseline_root = output_root / f"{timestamp}_dice_baseline"
    baseline_root.mkdir(parents=True, exist_ok=False)
    return baseline_root


def apply_limit(config: dict[str, Any], limit: int | None) -> dict[str, Any]:
    updated = dict(config)
    if limit is not None:
        updated["limit"] = limit
    return updated


def _timeout_value(timeout_seconds: int | None) -> int | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return timeout_seconds


def _flatten_summary(summary: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    return {
        "scenario": scenario_name,
        **{
            key: json.dumps(value, ensure_ascii=True) if isinstance(value, dict) else value
            for key, value in summary.items()
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_subprocess(
    *,
    command: list[str],
    timeout_seconds: int | None,
    cwd: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    timeout = _timeout_value(timeout_seconds)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "timeout_seconds": timeout,
            "runtime_seconds": time.monotonic() - started_at,
            "stdout": _output_text(exc.stdout),
            "stderr": _output_text(exc.stderr),
        }

    status = "completed" if completed.returncode == 0 else "failed"
    return {
        "status": status,
        "returncode": completed.returncode,
        "timeout_seconds": timeout,
        "runtime_seconds": time.monotonic() - started_at,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _list_directories(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {item for item in path.iterdir() if item.is_dir()}


def _single_new_directory(before: set[Path], after: set[Path]) -> Path | None:
    created = sorted(after - before)
    if len(created) == 1:
        return created[0]
    return created[-1] if created else None


def _failed_scenario_summary(
    *,
    scenario_name: str,
    status: str,
    reason: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario_name,
        "run_dir": "",
        "total_cases": 0,
        "feasible_cases": 0,
        "feasible_rate": 0.0,
        "mean_runtime_ms": 0.0,
        "mean_candidate_count": 0.0,
        "candidate_rows": 0,
        "target_success_rate": 0.0,
        "plausibility_pass_rate": 0.0,
        "immutable_violation_rate": 0.0,
        "mutable_violation_rate": 0.0,
        "bounds_violation_rate": 0.0,
        "directional_violation_rate": 0.0,
        "mean_lof_score": 0.0,
        "mean_distance_l1": 0.0,
        "mean_changed_feature_count": 0.0,
        "status_counts": json.dumps({status.upper(): 1}, ensure_ascii=True),
        "reason_counts": json.dumps({reason: 1}, ensure_ascii=True),
        "step_status": status,
        "step_reason": reason,
        "step_runtime_seconds": runtime_seconds,
    }


def run_baseline_scenarios(
    *,
    config_paths: list[Path],
    engine_config_path: Path,
    baseline_root: Path,
    scenario_limit: int | None,
    scenario_timeout_seconds: int | None,
) -> tuple[Path, list[dict[str, Any]]]:
    scenario_root = baseline_root / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]
    engine_config = load_config(engine_config_path)

    for config_path in config_paths:
        scenario_config = load_config(config_path)
        config = apply_limit(
            merge_configs(engine_config=engine_config, scenario_config=scenario_config),
            scenario_limit,
        )
        scenario_name = str(scenario_config.get("name") or config_path.stem)
        scenario_output_root = scenario_root / scenario_name
        scenario_output_root.mkdir(parents=True, exist_ok=False)
        effective_config_path = scenario_output_root / "effective_config.json"
        _write_json(effective_config_path, config)

        before = _list_directories(scenario_output_root)
        step = _run_subprocess(
            command=[
                sys.executable,
                str(repo_root / "experiments" / "scripts" / "run_benchmark.py"),
                "--config",
                str(effective_config_path),
                "--output-root",
                str(scenario_output_root),
            ],
            timeout_seconds=scenario_timeout_seconds,
            cwd=repo_root,
        )
        after = _list_directories(scenario_output_root)
        run_dir = _single_new_directory(before, after)
        step = {
            "step_type": "scenario",
            "scenario": scenario_name,
            "config_path": str(config_path),
            "effective_config_path": str(effective_config_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            **step,
        }
        step_results.append(step)

        if step["status"] == "completed" and run_dir is not None:
            summary = summarize_run(run_dir)
            write_summary_csv(run_dir / "summary.csv", summary)
            flattened = _flatten_summary(summary, scenario_name=scenario_name)
            flattened["step_status"] = step["status"]
            flattened["step_reason"] = ""
            flattened["step_runtime_seconds"] = step["runtime_seconds"]
            summary_rows.append(flattened)
        else:
            reason = "timeout" if step["status"] == "timeout" else "subprocess_failed"
            summary_rows.append(
                _failed_scenario_summary(
                    scenario_name=scenario_name,
                    status=str(step["status"]),
                    reason=reason,
                    runtime_seconds=float(step["runtime_seconds"]),
                )
            )

    _write_csv(scenario_root / "scenario_summary.csv", summary_rows)
    _write_json(
        scenario_root / "scenario_step_results.json",
        {"steps": step_results},
    )
    return scenario_root, step_results


def run_baseline_stability(
    *,
    baseline_root: Path,
    engine_config_path: Path,
    stability_config_path: Path,
    stability_limit: int | None,
    repeat_count: int,
    stability_timeout_seconds: int | None,
) -> tuple[Path | None, dict[str, Any]]:
    stability_parent = baseline_root / "stability"
    stability_parent.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[2]
    stability_config = apply_limit(
        merge_configs(
            engine_config=load_config(engine_config_path),
            scenario_config=load_config(stability_config_path),
        ),
        stability_limit,
    )
    effective_config_path = stability_parent / "effective_config.json"
    _write_json(effective_config_path, stability_config)

    before = _list_directories(stability_parent)
    step = _run_subprocess(
        command=[
            sys.executable,
            str(repo_root / "experiments" / "scripts" / "evaluate_stability.py"),
            "--config",
            str(effective_config_path),
            "--output-root",
            str(stability_parent),
            "--repeat-count",
            str(repeat_count),
        ],
        timeout_seconds=stability_timeout_seconds,
        cwd=repo_root,
    )
    after = _list_directories(stability_parent)
    stability_root = _single_new_directory(before, after)
    step = {
        "step_type": "stability",
        "config_path": str(stability_config_path),
        "effective_config_path": str(effective_config_path),
        "run_dir": str(stability_root) if stability_root is not None else None,
        **step,
    }
    _write_json(stability_parent / "stability_step_result.json", step)
    return stability_root if step["status"] == "completed" else None, step


def run_dice_baseline(
    *,
    output_root: Path,
    engine_config_path: Path,
    scenario_config_paths: list[Path],
    stability_config_path: Path,
    scenario_limit: int | None,
    stability_limit: int | None,
    repeat_count: int,
    skip_stability: bool,
    scenario_timeout_seconds: int | None = DEFAULT_SCENARIO_TIMEOUT_SECONDS,
    stability_timeout_seconds: int | None = DEFAULT_STABILITY_TIMEOUT_SECONDS,
) -> Path:
    baseline_root = _make_baseline_root(output_root)
    scenario_root, scenario_steps = run_baseline_scenarios(
        config_paths=scenario_config_paths,
        engine_config_path=engine_config_path,
        baseline_root=baseline_root,
        scenario_limit=scenario_limit,
        scenario_timeout_seconds=scenario_timeout_seconds,
    )

    stability_root: Path | None = None
    stability_step: dict[str, Any] | None = None
    if not skip_stability:
        stability_root, stability_step = run_baseline_stability(
            baseline_root=baseline_root,
            engine_config_path=engine_config_path,
            stability_config_path=stability_config_path,
            stability_limit=stability_limit,
            repeat_count=repeat_count,
            stability_timeout_seconds=stability_timeout_seconds,
        )

    collected = collect_results(baseline_root)
    manifest = {
        "baseline_root": str(baseline_root),
        "scenario_root": str(scenario_root),
        "stability_root": str(stability_root) if stability_root is not None else None,
        "engine_config": str(engine_config_path),
        "scenario_configs": [str(path) for path in scenario_config_paths],
        "stability_config": str(stability_config_path),
        "scenario_limit": scenario_limit,
        "stability_limit": stability_limit,
        "repeat_count": repeat_count,
        "skip_stability": skip_stability,
        "scenario_timeout_seconds": _timeout_value(scenario_timeout_seconds),
        "stability_timeout_seconds": _timeout_value(stability_timeout_seconds),
        "scenario_steps": scenario_steps,
        "stability_step": stability_step,
        "collected_outputs": {key: str(value) for key, value in collected.items()},
    }
    (baseline_root / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return baseline_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DiCE baseline experiment.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG_PATH)
    parser.add_argument(
        "--scenario-configs",
        nargs="*",
        type=Path,
        default=DEFAULT_SCENARIO_CONFIGS,
    )
    parser.add_argument("--stability-config", type=Path, default=DEFAULT_STABILITY_CONFIG)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--stability-limit", type=int, default=None)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument(
        "--scenario-timeout-seconds",
        type=int,
        default=DEFAULT_SCENARIO_TIMEOUT_SECONDS,
        help="Timeout per scenario subprocess. Use 0 to disable.",
    )
    parser.add_argument(
        "--stability-timeout-seconds",
        type=int,
        default=DEFAULT_STABILITY_TIMEOUT_SECONDS,
        help="Timeout for the stability subprocess. Use 0 to disable.",
    )
    args = parser.parse_args()

    output_dir = run_dice_baseline(
        output_root=args.output_root,
        engine_config_path=args.engine_config,
        scenario_config_paths=args.scenario_configs,
        stability_config_path=args.stability_config,
        scenario_limit=args.scenario_limit,
        stability_limit=args.stability_limit,
        repeat_count=args.repeat_count,
        skip_stability=args.skip_stability,
        scenario_timeout_seconds=args.scenario_timeout_seconds,
        stability_timeout_seconds=args.stability_timeout_seconds,
    )
    print(f"DiCE baseline output written to {output_dir}")


if __name__ == "__main__":
    main()
