import csv
import json
from pathlib import Path

from experiments.scripts.diagnose_ocean import (
    build_diagnostics_report,
    diagnose_engine,
    write_diagnostics,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_comparison(root: Path) -> None:
    dice_run = root / "baselines" / "dice" / "run-1" / "scenarios" / "all_mutable"
    ocean_run = root / "baselines" / "ocean" / "run-1" / "scenarios" / "all_mutable"
    dice_no_mutable_run = root / "baselines" / "dice" / "run-1" / "scenarios" / "no_mutable"
    ocean_no_mutable_run = root / "baselines" / "ocean" / "run-1" / "scenarios" / "no_mutable"
    ocean_activity_run = root / "baselines" / "ocean" / "run-1" / "scenarios" / "activity_only"
    _write_csv(
        root / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "run_dir": str(dice_run),
                "step_status": "completed",
                "feasible_rate": "1.0",
                "mean_runtime_ms": "100.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"OK": 2}',
            },
            {
                "scenario": "no_mutable",
                "run_dir": str(dice_no_mutable_run),
                "step_status": "completed",
                "feasible_rate": "0.0",
                "mean_runtime_ms": "5.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"NO_MUTABLE_FEATURE": 2}',
            },
            {
                "scenario": "activity_only",
                "run_dir": "",
                "step_status": "timeout",
                "feasible_rate": "0.0",
                "mean_runtime_ms": "0.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"timeout": 1}',
            },
        ],
    )
    _write_csv(
        root / "baselines" / "ocean" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "run_dir": str(ocean_run),
                "step_status": "completed",
                "feasible_rate": "0.5",
                "mean_runtime_ms": "300.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"OK": 1, "TARGET_UNREACHABLE_UNDER_CONSTRAINTS": 1}',
            },
            {
                "scenario": "no_mutable",
                "run_dir": str(ocean_no_mutable_run),
                "step_status": "completed",
                "feasible_rate": "0.0",
                "mean_runtime_ms": "5.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"NO_MUTABLE_FEATURE": 2}',
            },
            {
                "scenario": "activity_only",
                "run_dir": str(ocean_activity_run),
                "step_status": "completed",
                "feasible_rate": "0.0",
                "mean_runtime_ms": "200.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "bounds_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "reason_counts": '{"TARGET_UNREACHABLE_UNDER_CONSTRAINTS": 2}',
            },
        ],
    )
    _write_jsonl(
        dice_run / "cases.jsonl",
        [
            {"request_id": "exp-00001", "status": "FEASIBLE", "reason_code": "OK"},
            {"request_id": "exp-00002", "status": "FEASIBLE", "reason_code": "OK"},
        ],
    )
    _write_jsonl(
        ocean_run / "cases.jsonl",
        [
            {"request_id": "exp-00001", "status": "FEASIBLE", "reason_code": "OK"},
            {
                "request_id": "exp-00002",
                "status": "INFEASIBLE",
                "reason_code": "TARGET_UNREACHABLE_UNDER_CONSTRAINTS",
            },
        ],
    )
    _write_jsonl(
        dice_no_mutable_run / "cases.jsonl",
        [
            {
                "request_id": "exp-00001",
                "status": "INFEASIBLE",
                "reason_code": "NO_MUTABLE_FEATURE",
            },
            {
                "request_id": "exp-00002",
                "status": "INFEASIBLE",
                "reason_code": "NO_MUTABLE_FEATURE",
            },
        ],
    )
    _write_jsonl(
        ocean_no_mutable_run / "cases.jsonl",
        [
            {
                "request_id": "exp-00001",
                "status": "INFEASIBLE",
                "reason_code": "NO_MUTABLE_FEATURE",
            },
            {
                "request_id": "exp-00002",
                "status": "INFEASIBLE",
                "reason_code": "NO_MUTABLE_FEATURE",
            },
        ],
    )
    _write_jsonl(
        ocean_activity_run / "cases.jsonl",
        [
            {
                "request_id": "exp-00001",
                "status": "INFEASIBLE",
                "reason_code": "TARGET_UNREACHABLE_UNDER_CONSTRAINTS",
            },
            {
                "request_id": "exp-00002",
                "status": "INFEASIBLE",
                "reason_code": "TARGET_UNREACHABLE_UNDER_CONSTRAINTS",
            },
        ],
    )
    _write_csv(
        ocean_activity_run / "inputs.csv",
        [
            {
                "engine_name": "ocean",
                "request_id": "exp-00001",
                "baseline_probability_low_risk": "0.2",
                "baseline_probability_high_risk": "0.8",
                "features": '{"moderate_physical_activity_frequency": 0.0}',
            },
            {
                "engine_name": "ocean",
                "request_id": "exp-00002",
                "baseline_probability_low_risk": "0.1",
                "baseline_probability_high_risk": "0.9",
                "features": '{"moderate_physical_activity_frequency": 2.0}',
            },
        ],
    )
    _write_csv(
        ocean_run / "candidates.csv",
        [
            {
                "engine_name": "ocean",
                "request_id": "exp-00001",
                "candidate_id": "cf_1",
                "status": "FEASIBLE",
                "delta": '{"BMI": -2.0, "smoking_status": -1.0}',
            }
        ],
    )
    effective_config = {
        "expected_outcome": {
            "category": "expected_infeasible_control",
            "feasible": False,
            "reason_codes": ["NO_MUTABLE_FEATURE"],
        }
    }
    activity_config = {
        "mutable_allowed": ["moderate_physical_activity_frequency"],
        "feature_bounds": {"moderate_physical_activity_frequency": {"min": 0.0, "max": 14.0}},
    }
    (root / "baselines" / "dice" / "run-1" / "scenarios" / "no_mutable").mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "baselines" / "ocean" / "run-1" / "scenarios" / "no_mutable").mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        root / "baselines" / "dice" / "run-1" / "scenarios" / "no_mutable" / "effective_config.json"
    ).write_text(json.dumps(effective_config), encoding="utf-8")
    ocean_activity_run.mkdir(parents=True, exist_ok=True)
    (ocean_activity_run / "effective_config.json").write_text(
        json.dumps(activity_config),
        encoding="utf-8",
    )
    (
        root
        / "baselines"
        / "ocean"
        / "run-1"
        / "scenarios"
        / "no_mutable"
        / "effective_config.json"
    ).write_text(json.dumps(effective_config), encoding="utf-8")


def test_diagnose_engine_builds_ocean_gap_and_overlap(tmp_path: Path) -> None:
    _write_comparison(tmp_path)

    payload = diagnose_engine(tmp_path, target_engine="ocean", baseline_engine="dice")

    scenario = next(item for item in payload["scenarios"] if item["scenario"] == "all_mutable")
    assert payload["target_worse_than_baseline_scenarios"] == ["all_mutable"]
    assert payload["problematic_low_feasibility_scenarios"] == ["activity_only"]
    assert payload["expected_low_feasibility_scenarios"] == ["no_mutable"]

    activity = next(item for item in payload["scenarios"] if item["scenario"] == "activity_only")
    assert activity["constraint_diagnosis"]["input_count"] == 2
    assert activity["constraint_diagnosis"]["mutable_allowed"] == [
        "moderate_physical_activity_frequency"
    ]
    assert scenario["target_feasible_rate"] == 0.5
    assert scenario["baseline_feasible_rate"] == 1.0
    assert scenario["overlap_summary"]["both_feasible"] == 1
    assert scenario["overlap_summary"]["baseline_feasible_target_infeasible"] == 1
    assert scenario["target_top_changed_features"] == [
        {"feature": "BMI", "count": 1},
        {"feature": "smoking_status", "count": 1},
    ]


def test_build_diagnostics_report_contains_main_tables(tmp_path: Path) -> None:
    _write_comparison(tmp_path)
    payload = diagnose_engine(tmp_path, target_engine="ocean", baseline_engine="dice")

    report = build_diagnostics_report(payload)

    assert "# OCEAN Diagnostics" in report
    assert "| all_mutable | observed_feasible | completed | 50.0% | 100.0% | -50.0%" in report
    assert "Expected low feasibility controls: no_mutable" in report
    assert "Problematic low feasibility scenarios: activity_only" in report
    assert "Constraint/Search-Space Diagnostics" in report
    assert "moderate_physical_activity_frequency" in report
    assert "TARGET_UNREACHABLE_UNDER_CONSTRAINTS" in report
    assert "| BMI | 1 |" in report


def test_no_mutable_reason_is_expected_control_without_metadata(tmp_path: Path) -> None:
    _write_comparison(tmp_path)
    (
        tmp_path
        / "baselines"
        / "ocean"
        / "run-1"
        / "scenarios"
        / "no_mutable"
        / "effective_config.json"
    ).unlink()

    payload = diagnose_engine(tmp_path, target_engine="ocean", baseline_engine="dice")

    assert payload["problematic_low_feasibility_scenarios"] == ["activity_only"]
    assert payload["expected_low_feasibility_scenarios"] == ["no_mutable"]


def test_write_diagnostics_creates_report_and_json(tmp_path: Path) -> None:
    _write_comparison(tmp_path)
    payload = diagnose_engine(tmp_path, target_engine="ocean", baseline_engine="dice")

    outputs = write_diagnostics(tmp_path, payload)

    assert outputs["report"] == tmp_path / "ocean_diagnostics.md"
    assert outputs["json"] == tmp_path / "ocean_diagnostics.json"
    assert outputs["report"].exists()
    assert outputs["json"].exists()


def test_write_diagnostics_uses_target_engine_name_for_output_files(tmp_path: Path) -> None:
    _write_comparison(tmp_path)
    payload = diagnose_engine(tmp_path, target_engine="ft", baseline_engine="dice")

    outputs = write_diagnostics(tmp_path, payload)

    assert outputs["report"] == tmp_path / "ft_diagnostics.md"
    assert outputs["json"] == tmp_path / "ft_diagnostics.json"
