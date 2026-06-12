import csv
import json
from pathlib import Path

from experiments.scripts.run_comparison import (
    build_comparison_report,
    engine_from_source_file,
    write_comparison_report,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_engine_from_source_file_reads_baseline_layout() -> None:
    source_file = str(
        Path("experiments/results/comparisons/20260501_000000")
        / "baselines"
        / "dice"
        / "20260501_000001"
        / "combined"
        / "scenario_summary.csv"
    )

    assert engine_from_source_file(source_file) == "dice"


def test_build_comparison_report_contains_engine_summary(tmp_path: Path) -> None:
    (tmp_path / "baselines" / "dice" / "run-1" / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "engine": "dice",
                "scenario_steps": [{"scenario": "all_mutable", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "target_success_rate_all_candidates": "1.0",
                "plausibility_pass_rate": "1.0",
                "mean_lof_score": "1.0",
                "mean_changed_feature_count": "1.0",
                "mean_distance_l1": "0.2",
                "mean_runtime_ms": "12.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
            }
        ],
    )
    _write_csv(
        tmp_path
        / "baselines"
        / "dice"
        / "run-1"
        / "stability"
        / "run-1"
        / "stability_aggregate.csv",
        [
            {
                "mean_feasible_only_jaccard_changed_features": "1.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        tmp_path
        / "baselines"
        / "dice"
        / "run-1"
        / "scenarios"
        / "all_mutable"
        / "run-1"
        / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": '{"BMI": -1.0}'}],
    )

    report = build_comparison_report(tmp_path)

    assert "# Comparison Report" in report
    assert "| dice | 1 | 1 | 0 | 0 | 100.0% | 1 |" in report
    assert "Scenario Matrix" in report
    assert "Successful-only Jaccard" in report
    assert "BMI" in report


def test_write_comparison_report_creates_report_file(tmp_path: Path) -> None:
    (tmp_path / "baselines" / "dice" / "run-1" / "baseline_manifest.json").write_text(
        json.dumps({"engine": "dice", "scenario_steps": []}),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [{"scenario": "all_mutable", "target_success_rate_all_candidates": "1.0"}],
    )

    report_path = write_comparison_report(tmp_path)

    assert report_path == tmp_path / "comparison_report.md"
    assert report_path.exists()
