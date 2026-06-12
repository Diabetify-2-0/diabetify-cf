import csv
import json
from pathlib import Path

from experiments.scripts.print_baseline_report import (
    _scenario_rows,
    _top_changed_features,
    build_markdown_report,
    write_markdown_report,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_scenario_rows_can_read_partial_baseline_output(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "scenarios" / "all_mutable" / "run-1" / "summary.csv",
        [{"target_success_rate_all_candidates": "1.0", "plausibility_pass_rate": "1.0"}],
    )

    rows = _scenario_rows(tmp_path)

    assert rows[0]["scenario"] == "all_mutable"
    assert rows[0]["target_success_rate_all_candidates"] == "1.0"


def test_top_changed_features_counts_delta_keys() -> None:
    rows = [
        {"delta": '{"BMI": -1.0, "is_hypertension": -1.0}'},
        {"delta": '{"BMI": -2.0}'},
    ]

    top_features = _top_changed_features(rows, top_n=2)

    assert top_features == [("BMI", 2), ("is_hypertension", 1)]


def test_build_markdown_report_contains_scenario_and_stability_sections(tmp_path: Path) -> None:
    (tmp_path / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "engine": "dice",
                "scenario_steps": [{"scenario": "all_mutable", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "target_success_rate_all_candidates": "1.0",
                "plausibility_pass_rate": "1.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "mean_changed_feature_count": "1.0",
                "mean_distance_l1": "0.2",
                "mean_runtime_ms": "12.0",
                "mean_lof_score": "1.0",
            }
        ],
    )
    _write_csv(
        tmp_path / "stability" / "run-1" / "stability_aggregate.csv",
        [
            {
                "mean_feasible_only_jaccard_changed_features": "1.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        tmp_path / "scenarios" / "all_mutable" / "run-1" / "candidates.csv",
        [{"delta": '{"BMI": -1.0}'}],
    )

    report = build_markdown_report(tmp_path)

    assert "# Baseline Report" in report
    assert "all_mutable" in report
    assert "Successful-only Jaccard" in report
    assert "BMI" in report


def test_write_markdown_report_creates_report_file(tmp_path: Path) -> None:
    (tmp_path / "baseline_manifest.json").write_text(
        json.dumps({"engine": "dice", "scenario_steps": []}),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "combined_scenario_summary.csv",
        [{"scenario": "all_mutable", "target_success_rate_all_candidates": "1.0"}],
    )

    report_path = write_markdown_report(tmp_path)

    assert report_path == tmp_path / "report.md"
    assert report_path.exists()
