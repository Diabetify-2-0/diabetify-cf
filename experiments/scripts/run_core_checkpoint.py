from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError: 
    from _bootstrap import bootstrap_path


REPO_ROOT = bootstrap_path(__file__)
DEFAULT_SCOPE_PATH = REPO_ROOT / "experiments" / "configs" / "benchmark_scope" / "core_benchmark.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_LATEST_COMPARISON = DEFAULT_RESULTS_ROOT / "latest" / "comparison.txt"


def load_scope(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Benchmark scope file must contain a JSON object.")
    core_engines = data.get("core_engines")
    if not isinstance(core_engines, list) or not core_engines:
        raise ValueError("Benchmark scope file must define at least one core engine.")
    return data


def core_engine_config_paths(scope: dict[str, Any]) -> list[Path]:
    config_paths: list[Path] = []
    for entry in scope["core_engines"]:
        config_path = entry.get("config_path")
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("Each core engine entry must define a non-empty config_path.")
        config_paths.append(REPO_ROOT / config_path)
    return config_paths


def core_engine_names(scope: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in scope["core_engines"]:
        name = entry.get("engine")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each core engine entry must define a non-empty engine name.")
        names.append(name.strip().lower())
    return names


def build_checkpoint_command(
    *,
    scope: dict[str, Any],
    output_root: Path,
    scenario_limit: int,
    stability_limit: int,
    repeat_count: int,
    scenario_timeout_seconds: int,
    stability_timeout_seconds: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_checkpoint.py"),
        "--output-root",
        str(output_root),
        "--engine-configs",
    ]
    command.extend(str(path) for path in core_engine_config_paths(scope))
    command.extend(
        [
            "--scenario-limit",
            str(scenario_limit),
            "--stability-limit",
            str(stability_limit),
            "--repeat-count",
            str(repeat_count),
            "--scenario-timeout-seconds",
            str(scenario_timeout_seconds),
            "--stability-timeout-seconds",
            str(stability_timeout_seconds),
            "--required-engines",
        ]
    )
    command.extend(core_engine_names(scope))
    return command


def latest_comparison_pointer(output_root: Path) -> Path:
    return output_root / "latest" / "comparison.txt"


def latest_comparison_root(latest_pointer: Path) -> Path:
    comparison_root = latest_pointer.read_text(encoding="utf-8").strip()
    if not comparison_root:
        raise ValueError("Latest comparison pointer is empty.")
    return REPO_ROOT / comparison_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked core benchmark checkpoint.")
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--scenario-limit", type=int, default=10)
    parser.add_argument("--stability-limit", type=int, default=10)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--scenario-timeout-seconds", type=int, default=180)
    parser.add_argument("--stability-timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    scope = load_scope(args.scope)
    command = build_checkpoint_command(
        scope=scope,
        output_root=args.output_root,
        scenario_limit=args.scenario_limit,
        stability_limit=args.stability_limit,
        repeat_count=args.repeat_count,
        scenario_timeout_seconds=args.scenario_timeout_seconds,
        stability_timeout_seconds=args.stability_timeout_seconds,
    )
    subprocess.run(command, check=True, cwd=REPO_ROOT)

    comparison_root = latest_comparison_root(latest_comparison_pointer(args.output_root))
    print(f"Core benchmark comparison root: {comparison_root.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
