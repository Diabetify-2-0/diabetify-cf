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

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:  
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.run_benchmark import ( 
    DEFAULT_ENGINE_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    load_config,
    merge_configs,
)
from experiments.scripts.run_metadata import build_run_metadata 
from experiments.scripts.summarize_results import summarize_run, write_summary_csv 

DEFAULT_SCENARIO_CONFIGS = [
    Path("experiments/configs/scenarios/all_mutable.json"),
    Path("experiments/configs/scenarios/lifestyle_combo.json"),
    Path("experiments/configs/scenarios/bmi_only.json"),
    Path("experiments/configs/scenarios/activity_only.json"),
    Path("experiments/configs/scenarios/tight_bounds.json"),
    Path("experiments/configs/scenarios/no_mutable.json"),
]
DEFAULT_SCENARIO_TIMEOUT_SECONDS = 300


def _make_output_root(base_output_root: Path, engine_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = base_output_root / "scenarios" / engine_name / timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


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


def _timeout_value(timeout_seconds: int | None) -> int | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return timeout_seconds


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


def _single_new_run_directory(before: set[Path], after: set[Path], root: Path) -> Path | None:
    candidates = [path.parent for path in root.rglob("cases.jsonl")]
    if candidates:
        return sorted(candidates)[-1]
    return _single_new_directory(before, after)


def _apply_limit(config: dict[str, Any], limit: int | None) -> dict[str, Any]:
    updated = dict(config)
    if limit is not None:
        updated["limit"] = limit
    return updated


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


def _scenario_name(config: dict[str, Any], config_path: Path) -> str:
    return str(config.get("name") or config_path.stem)


def run_scenarios(
    config_paths: list[Path],
    output_root: Path,
    engine_config_path: Path,
    scenario_limit: int | None = None,
    timeout_seconds: int | None = DEFAULT_SCENARIO_TIMEOUT_SECONDS,
) -> Path:
    summary_rows: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []
    engine_config = load_config(engine_config_path)
    engine_name = str(engine_config.get("engine", "dice"))
    scenario_root = _make_output_root(output_root, engine_name)

    for config_path in config_paths:
        scenario_config = load_config(config_path)
        config = _apply_limit(
            merge_configs(engine_config=engine_config, scenario_config=scenario_config),
            scenario_limit,
        )
        scenario_name = _scenario_name(scenario_config, config_path)
        scenario_output_root = scenario_root / scenario_name
        scenario_output_root.mkdir(parents=True, exist_ok=False)
        effective_config_path = scenario_output_root / "effective_config.json"
        _write_json(effective_config_path, config)

        print(f"Running scenario '{scenario_name}'...", flush=True)
        before = _list_directories(scenario_output_root)
        step = _run_subprocess(
            command=[
                sys.executable,
                str(REPO_ROOT / "experiments" / "scripts" / "run_benchmark.py"),
                "--config",
                str(effective_config_path),
                "--output-root",
                str(scenario_output_root),
            ],
            timeout_seconds=timeout_seconds,
            cwd=REPO_ROOT,
        )
        after = _list_directories(scenario_output_root)
        run_dir = _single_new_run_directory(before, after, scenario_output_root)
        step = {
            "step_type": "scenario",
            "scenario": scenario_name,
            "config_path": str(config_path),
            "effective_config_path": str(effective_config_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            **step,
        }
        step_results.append(step)
        print(
            f"Scenario '{scenario_name}' finished with status={step['status']}.",
            flush=True,
        )

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

    metadata = build_run_metadata(
        repo_root=REPO_ROOT,
        run_type="scenarios",
        engine_name=engine_name,
        config_path=None,
    )
    metadata["engine_config"] = str(engine_config_path)
    metadata["scenario_configs"] = [str(path) for path in config_paths]
    metadata["scenario_limit"] = scenario_limit
    metadata["scenario_timeout_seconds"] = _timeout_value(timeout_seconds)
    (scenario_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_csv(scenario_root / "scenario_summary.csv", summary_rows)
    _write_json(scenario_root / "scenario_step_results.json", {"steps": step_results})
    return scenario_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple benchmark scenarios.")
    parser.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=DEFAULT_SCENARIO_CONFIGS,
        help="Scenario config files to run.",
    )
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SCENARIO_TIMEOUT_SECONDS,
        help="Timeout per scenario subprocess. Use 0 to disable.",
    )
    args = parser.parse_args()

    output_dir = run_scenarios(
        config_paths=args.configs,
        output_root=args.output_root,
        engine_config_path=args.engine_config,
        scenario_limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Scenario outputs written to {output_dir}")


if __name__ == "__main__":
    main()
