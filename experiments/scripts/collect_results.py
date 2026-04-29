from __future__ import annotations

import argparse
import csv
import sys
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

from experiments.scripts.run_benchmark import DEFAULT_OUTPUT_ROOT  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def collect_scenario_summaries(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("scenario_summary.csv")):
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_stability_summaries(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("stability_aggregate.csv")):
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_candidates(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("candidates.csv")):
        for row in _read_csv(path):
            rows.append({"source_file": str(path), **row})
    return rows


def collect_results(results_root: Path) -> dict[str, Path]:
    scenario_rows = collect_scenario_summaries(results_root)
    stability_rows = collect_stability_summaries(results_root)
    candidate_rows = collect_candidates(results_root)

    scenario_path = results_root / "combined_scenario_summary.csv"
    stability_path = results_root / "combined_stability_summary.csv"
    candidate_path = results_root / "combined_candidates.csv"
    _write_csv(scenario_path, scenario_rows)
    _write_csv(stability_path, stability_rows)
    _write_csv(candidate_path, candidate_rows)
    return {
        "scenario_summary": scenario_path,
        "stability_summary": stability_path,
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
