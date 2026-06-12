import csv
import json
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
    (root / "baselines" / "dice" / "run-1" / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "engine": "dice",
                "scenario_steps": [{"scenario": "all_mutable", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "target_success_rate_all_candidates": "1.0",
                "plausibility_pass_rate": "1.0",
                "mean_lof_score": "1.0",
                "mean_changed_feature_count": "1.0",
                "mean_distance_l1": "0.1",
                "mean_runtime_ms": "12.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "stability" / "run-1" / "stability_aggregate.csv",
        [
            {
                "mean_feasible_only_jaccard_changed_features": "1.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "scenarios" / "all_mutable" / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": "{}"}],
    )


def test_audit_comparison_passes_for_single_required_engine(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice"],
    )

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["scenario_summary"]["dice"]["completed_count"] == 1
    assert (tmp_path / "audit_report.json").exists()


def test_audit_comparison_fails_when_required_engine_is_missing(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice", "nn"],
    )

    assert payload["ok"] is False
    assert "Missing required engine rows: nn" in payload["errors"]


def test_audit_comparison_fails_on_failed_scenario_status(tmp_path: Path) -> None:
    _write_minimal_comparison(tmp_path)
    (tmp_path / "baselines" / "dice" / "run-1" / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "engine": "dice",
                "scenario_steps": [{"scenario": "all_mutable", "status": "failed"}],
            }
        ),
        encoding="utf-8",
    )

    payload = audit_comparison(
        tmp_path,
        required_engines=["dice"],
    )

    assert payload["ok"] is False
    assert "dice/all_mutable failed." in payload["errors"]


def test_latest_comparison_root_reads_pointer(tmp_path: Path) -> None:
    comparison_root = tmp_path / "comparisons" / "run-1"
    pointer = tmp_path / "latest" / "comparison.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(comparison_root), encoding="utf-8")

    assert _latest_comparison_root(tmp_path) == comparison_root


def test_latest_comparison_root_fails_without_pointer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _latest_comparison_root(tmp_path)
