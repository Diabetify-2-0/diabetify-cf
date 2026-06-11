import json
from pathlib import Path

from experiments.scripts.run_checkpoint import (
    build_checkpoint_report,
    write_checkpoint_manifest,
    write_checkpoint_report,
)


def _audit_payload(ok: bool = True) -> dict:
    return {
        "comparison_root": "experiments/results/comparisons/run-1",
        "ok": ok,
        "errors": [] if ok else ["dice/all_mutable failed."],
        "warnings": ["dice/bmi_only timed out."],
        "required_engines": ["dice"],
        "observed_engines": ["dice"],
        "row_counts": {"scenario": 6, "stability": 1, "candidates": 35},
        "scenario_summary": {
            "dice": {
                "scenario_count": 6,
                "completed_count": 3,
                "timeout_count": 3,
                "failed_count": 0,
                "mean_feasible_rate": 0.3333,
                "max_violation_rate": 0.0,
            },
        },
        "stability_summary": {
            "dice": {
                "case_count": 5,
                "mean_feasible_rate": 1.0,
                "fully_feasible_case_rate": 1.0,
                "stability_evaluable_case_rate": 1.0,
                "mean_feasible_only_jaccard_changed_features": 0.8667,
                "mean_feasible_only_stability_std_norm": 0.0571,
            },
        },
    }


def test_build_checkpoint_report_contains_audit_summary() -> None:
    report = build_checkpoint_report(_audit_payload())

    assert "# Experiment Checkpoint" in report
    assert "- Status: `PASS`" in report
    assert "| dice | 6 | 3 | 3 | 0 | 33.3% | 0.0000 |" in report
    assert "dice/bmi_only timed out." in report


def test_build_checkpoint_report_includes_errors_on_failure() -> None:
    report = build_checkpoint_report(_audit_payload(ok=False))

    assert "- Status: `FAIL`" in report
    assert "dice/all_mutable failed." in report


def test_write_checkpoint_report_creates_markdown(tmp_path: Path) -> None:
    report_path = write_checkpoint_report(tmp_path, _audit_payload())

    assert report_path == tmp_path / "checkpoint_report.md"
    assert "Experiment Checkpoint" in report_path.read_text(encoding="utf-8")


def test_write_checkpoint_manifest_creates_json(tmp_path: Path) -> None:
    manifest_path = write_checkpoint_manifest(
        tmp_path,
        audit_payload=_audit_payload(),
        checkpoint_config={"scenario_limit": 5},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["checkpoint_config"] == {"scenario_limit": 5}
    assert payload["checkpoint_report"] == str(tmp_path / "checkpoint_report.md")
