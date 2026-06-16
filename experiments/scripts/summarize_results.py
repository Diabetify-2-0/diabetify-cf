from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _positive_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if float(row.get(key, 0) or 0) > 0) / len(rows)


def _boolean_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _truthy(row.get(key))) / len(rows)


def _boolean_rate_over_total_cases(
    rows: list[dict[str, Any]],
    key: str,
    total_cases: int,
) -> float:
    if total_cases <= 0:
        return 0.0
    return sum(1 for row in rows if _truthy(row.get(key))) / total_cases


def summarize_run(run_dir: Path) -> dict[str, Any]:
    candidates = _read_csv(run_dir / "candidates.csv")
    return {
        "target_success_rate_all_candidates": _boolean_rate(candidates, "target_success"),
        "plausibility_pass_rate": _boolean_rate(candidates, "plausibility_pass"),
        "mean_lof_score": _mean_float(candidates, "lof_score"),
        "mean_changed_feature_count": _mean_float(candidates, "changed_feature_count"),
        "mean_distance_l1": _mean_float(candidates, "distance_l1"),
        "mean_runtime_ms": _mean_float(_read_jsonl(run_dir / "cases.jsonl"), "runtime_ms"),
        "immutable_violation_rate": _positive_rate(candidates, "immutable_violation_count"),
        "mutable_violation_rate": _positive_rate(candidates, "mutable_violation_count"),
        "directional_violation_rate": _positive_rate(candidates, "directional_violation_count"),
        "transition_violation_rate": _positive_rate(candidates, "transition_violation_count"),
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    flat = {
        key: json.dumps(value, ensure_ascii=True) if isinstance(value, dict) else value
        for key, value in summary.items()
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a benchmark run directory.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    write_summary_csv(args.run_dir / "summary.csv", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
