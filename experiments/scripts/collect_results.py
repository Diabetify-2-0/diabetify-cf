from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:  
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.run_benchmark import DEFAULT_OUTPUT_ROOT 


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_generated_combined_path(path: Path) -> bool:
    return "combined" in path.parts or path.name.startswith("combined_")


def collect_scenario_summaries(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("scenario_summary.csv")):
        if _is_generated_combined_path(path):
            continue
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_stability_summaries(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("stability_aggregate.csv")):
        if _is_generated_combined_path(path):
            continue
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_candidates(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("candidates.csv")):
        if _is_generated_combined_path(path):
            continue
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_inputs(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("inputs.csv")):
        if _is_generated_combined_path(path):
            continue
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_results(results_root: Path) -> dict[str, Path]:
    scenario_rows = collect_scenario_summaries(results_root)
    stability_rows = collect_stability_summaries(results_root)
    input_rows = collect_inputs(results_root)
    candidate_rows = collect_candidates(results_root)

    combined_root = results_root / "combined"
    scenario_path = combined_root / "scenario_summary.csv"
    stability_path = combined_root / "stability_summary.csv"
    input_path = combined_root / "inputs.csv"
    candidate_path = combined_root / "candidates.csv"
    _write_csv(scenario_path, scenario_rows)
    _write_csv(stability_path, stability_rows)
    _write_csv(input_path, input_rows)
    _write_csv(candidate_path, candidate_rows)
    return {
        "scenario_summary": scenario_path,
        "stability_summary": stability_path,
        "inputs": input_path,
        "candidates": candidate_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment summary files.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    outputs = collect_results(args.results_root)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
