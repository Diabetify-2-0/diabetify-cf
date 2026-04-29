from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def summarize_run(run_dir: Path) -> dict[str, Any]:
    cases = _read_jsonl(run_dir / "cases.jsonl")
    candidates = _read_csv(run_dir / "candidates.csv")
    status_counts = Counter(str(row.get("status", "UNKNOWN")) for row in cases)
    reason_counts = Counter(str(row.get("reason_code", "UNKNOWN")) for row in cases)
    runtimes = [float(row["runtime_ms"]) for row in cases if row.get("runtime_ms") is not None]
    candidate_counts = [
        int(row["candidate_count"]) for row in cases if row.get("candidate_count") is not None
    ]

    total_cases = len(cases)
    feasible_cases = status_counts.get("FEASIBLE", 0)
    return {
        "run_dir": str(run_dir),
        "total_cases": total_cases,
        "feasible_cases": feasible_cases,
        "feasible_rate": feasible_cases / total_cases if total_cases else 0.0,
        "mean_runtime_ms": sum(runtimes) / len(runtimes) if runtimes else 0.0,
        "mean_candidate_count": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "candidate_rows": len(candidates),
        "target_success_rate": _boolean_rate(candidates, "target_success"),
        "plausibility_pass_rate": _boolean_rate(candidates, "plausibility_pass"),
        "immutable_violation_rate": _positive_rate(candidates, "immutable_violation_count"),
        "mutable_violation_rate": _positive_rate(candidates, "mutable_violation_count"),
        "bounds_violation_rate": _positive_rate(candidates, "bounds_violation_count"),
        "directional_violation_rate": _positive_rate(candidates, "directional_violation_count"),
        "mean_lof_score": _mean_float(candidates, "lof_score"),
        "mean_distance_l1": _mean_float(candidates, "distance_l1"),
        "mean_changed_feature_count": _mean_float(candidates, "changed_feature_count"),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
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
