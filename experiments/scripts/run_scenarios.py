from __future__ import annotations

import argparse
import csv
import json
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

from experiments.scripts.run_benchmark import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    load_config,
    merge_configs,
    run_benchmark,
)
from experiments.scripts.run_metadata import build_run_metadata  # noqa: E402
from experiments.scripts.summarize_results import summarize_run, write_summary_csv  # noqa: E402

DEFAULT_SCENARIO_CONFIGS = [
    Path("experiments/configs/scenarios/all_mutable.json"),
    Path("experiments/configs/scenarios/bmi_only.json"),
    Path("experiments/configs/scenarios/activity_only.json"),
    Path("experiments/configs/scenarios/tight_bounds.json"),
    Path("experiments/configs/scenarios/no_mutable.json"),
]


def _make_output_root(base_output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = base_output_root / f"{timestamp}_scenarios"
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


def _scenario_name(config: dict[str, Any], config_path: Path) -> str:
    return str(config.get("name") or config_path.stem)


def run_scenarios(config_paths: list[Path], output_root: Path, engine_config_path: Path) -> Path:
    scenario_root = _make_output_root(output_root)
    summary_rows: list[dict[str, Any]] = []
    engine_config = load_config(engine_config_path)

    for config_path in config_paths:
        scenario_config = load_config(config_path)
        config = merge_configs(engine_config=engine_config, scenario_config=scenario_config)
        scenario_name = _scenario_name(scenario_config, config_path)
        scenario_output_root = scenario_root / scenario_name
        run_dir = run_benchmark(
            config=config,
            output_root=scenario_output_root,
            config_path=config_path,
        )
        summary = summarize_run(run_dir)
        write_summary_csv(run_dir / "summary.csv", summary)
        summary_rows.append(_flatten_summary(summary, scenario_name=scenario_name))

    engine_name = str(engine_config.get("engine", "dice"))
    metadata = build_run_metadata(
        repo_root=REPO_ROOT,
        run_type="scenarios",
        engine_name=engine_name,
        config_path=None,
    )
    metadata["engine_config"] = str(engine_config_path)
    metadata["scenario_configs"] = [str(path) for path in config_paths]
    (scenario_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_csv(scenario_root / "scenario_summary.csv", summary_rows)
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
    args = parser.parse_args()

    output_dir = run_scenarios(
        config_paths=args.configs,
        output_root=args.output_root,
        engine_config_path=args.engine_config,
    )
    print(f"Scenario outputs written to {output_dir}")


if __name__ == "__main__":
    main()
