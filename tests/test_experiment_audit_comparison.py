import csv
from pathlib import Path

import pytest

from experiments.scripts.audit_comparison import (
    _latest_comparison_root,
    audit_comparison,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_minimal_comparison(root: Path) -> None:
    (root / "comparison_manifest.json").write_text("{}", encoding="utf-8")
    (root / "comparison_report.md").write_text("# Comparison Report\n", encoding="utf-8")
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "completed",
                "feasible_rate": "1.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "ocean" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "timeout",
                "feasible_rate": "0.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "stability" / "run-1" / "stability_aggregate.csv",
        [
            {
                "case_count": "1",
                "mean_feasible_rate": "1.0",
                "fully_feasible_case_rate": "1.0",
                "stability_evaluable_case_rate": "1.0",
                "mean_feasible_only_jaccard_changed_features": "1.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "ocean" / "run-1" / "stability" / "run-1" / "stability_aggregate.csv",
        [
            {
                "case_count": "1",
                "mean_feasible_rate": "0.0",
                "fully_feasible_case_rate": "0.0",
                "stability_evaluable_case_rate": "0.0",
                "mean_feasible_only_jaccard_changed_features": "0.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "scenarios" / "all_mutable" / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": "{}"}],
    )


def test_audit_comparison_passes_with_timeout_warning(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice", "ocean"],
        max_allowed_violation_rate=0.0,
    )

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert "ocean/all_mutable timed out." in payload["warnings"]
    assert payload["scenario_summary"]["dice"]["completed_count"] == 1
    assert (tmp_path / "audit_report.json").exists()


def test_audit_comparison_can_fail_on_timeout(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice", "ocean"],
        max_allowed_violation_rate=0.0,
        fail_on_timeout=True,
    )

    assert payload["ok"] is False
    assert "ocean/all_mutable timed out." in payload["errors"]


def test_audit_comparison_fails_on_constraint_violation(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)
    _write_csv(
        tmp_path / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "completed",
                "feasible_rate": "1.0",
                "immutable_violation_rate": "0.1",
                "mutable_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
            }
        ],
    )

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice", "ocean"],
        max_allowed_violation_rate=0.0,
    )

    assert payload["ok"] is False
    assert any("immutable_violation_rate" in error for error in payload["errors"])


def test_latest_comparison_root_reads_pointer(tmp_path: Path) -> None:
    comparison_root = tmp_path / "comparisons" / "run-1"
    pointer = tmp_path / "latest" / "comparison.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(comparison_root), encoding="utf-8")

    assert _latest_comparison_root(tmp_path) == comparison_root


def test_latest_comparison_root_fails_without_pointer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _latest_comparison_root(tmp_path)
