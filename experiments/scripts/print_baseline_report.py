from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_path()

from experiments.scripts.collect_results import collect_results  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def _scenario_from_summary_path(path: Path) -> str:
    parts = path.parts
    if "scenarios" in parts:
        index = parts.index("scenarios")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def _scenario_rows(baseline_root: Path) -> list[dict[str, str]]:
    combined = baseline_root / "combined_scenario_summary.csv"
    if combined.exists():
        rows = _read_csv(combined)
        if rows:
            return rows

    scenario_summary = baseline_root / "scenarios" / "scenario_summary.csv"
    if scenario_summary.exists():
        rows = _read_csv(scenario_summary)
        if rows:
            return rows

    rows: list[dict[str, str]] = []
    for path in sorted((baseline_root / "scenarios").rglob("summary.csv")):
        for row in _read_csv(path):
            rows.append({"scenario": _scenario_from_summary_path(path), **row})
    return rows


def _stability_rows(baseline_root: Path) -> list[dict[str, str]]:
    combined = baseline_root / "combined_stability_summary.csv"
    if combined.exists():
        rows = _read_csv(combined)
        if rows:
            return rows
    return [
        row
        for path in sorted(baseline_root.rglob("stability_aggregate.csv"))
        for row in _read_csv(path)
    ]


def _candidate_rows(baseline_root: Path) -> list[dict[str, str]]:
    combined = baseline_root / "combined_candidates.csv"
    if combined.exists():
        rows = _read_csv(combined)
        if rows:
            return rows
    return [
        row for path in sorted(baseline_root.rglob("candidates.csv")) for row in _read_csv(path)
    ]


def _top_changed_features(candidates: list[dict[str, str]], top_n: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in candidates:
        raw_delta = row.get("delta", "{}")
        try:
            delta = json.loads(raw_delta)
        except json.JSONDecodeError:
            continue
        if not isinstance(delta, dict):
            continue
        for feature_name in delta:
            counts[str(feature_name)] += 1
    return counts.most_common(top_n)


def _print_scenarios(rows: list[dict[str, str]]) -> None:
    print("\nScenario Summary")
    if not rows:
        print("  No scenario summary rows found.")
        return

    header = (
        "scenario",
        "status",
        "feasible",
        "target",
        "immutable",
        "mutable",
        "bounds",
        "direction",
        "runtime_ms",
        "lof",
    )
    print("  " + " | ".join(header))
    print("  " + "-" * 104)
    for row in rows:
        print(
            "  "
            + " | ".join(
                [
                    row.get("scenario", "-"),
                    row.get("step_status") or "completed",
                    _percent(_as_float(row, "feasible_rate")),
                    _percent(_as_float(row, "target_success_rate")),
                    _percent(_as_float(row, "immutable_violation_rate")),
                    _percent(_as_float(row, "mutable_violation_rate")),
                    _percent(_as_float(row, "bounds_violation_rate")),
                    _percent(_as_float(row, "directional_violation_rate")),
                    _number(_as_float(row, "mean_runtime_ms")),
                    _number(_as_float(row, "mean_lof_score")),
                ]
            )
        )


def _print_stability(rows: list[dict[str, str]]) -> None:
    print("\nStability Summary")
    if not rows:
        print("  No stability summary rows found.")
        return

    for row in rows:
        print(
            "  "
            f"case_count={row.get('case_count', '0')} | "
            f"mean_feasible_rate={_percent(_as_float(row, 'mean_feasible_rate'))} | "
            "mean_jaccard_changed_features="
            f"{_number(_as_float(row, 'mean_jaccard_changed_features'))} | "
            f"mean_stability_std_norm={_number(_as_float(row, 'mean_stability_std_norm'))}"
        )


def _print_top_features(candidates: list[dict[str, str]], top_n: int) -> None:
    print("\nTop Changed Features")
    top_features = _top_changed_features(candidates, top_n)
    if not top_features:
        print("  No candidate deltas found.")
        return
    for feature_name, count in top_features:
        print(f"  {feature_name}: {count}")


def print_report(baseline_root: Path, top_n: int) -> None:
    if not baseline_root.exists():
        raise FileNotFoundError(f"Baseline root not found: {baseline_root}")

    collect_results(baseline_root)
    print(f"Baseline root: {baseline_root}")
    _print_scenarios(_scenario_rows(baseline_root))
    _print_stability(_stability_rows(baseline_root))
    _print_top_features(_candidate_rows(baseline_root), top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a readable DiCE baseline report.")
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    print_report(args.baseline_root, top_n=args.top_n)


if __name__ == "__main__":
    main()
