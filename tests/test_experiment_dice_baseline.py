import csv
import json
from pathlib import Path

from experiments.scripts.run_baseline import apply_limit, run_baseline_scenarios


def test_apply_limit_overrides_limit_when_present() -> None:
    config = {"engine": "dice", "limit": 10}

    updated = apply_limit(config, 3)

    assert updated["limit"] == 3
    assert config["limit"] == 10


def test_apply_limit_keeps_config_when_limit_is_none() -> None:
    config = {"engine": "dice", "limit": 10}

    updated = apply_limit(config, None)

    assert updated == config
    assert updated is not config


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_baseline_scenarios_records_completed_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "dice_case.json"
    config_path.write_text(json.dumps({"engine": "dice", "limit": 10}), encoding="utf-8")
    engine_config_path = tmp_path / "engine.json"
    engine_config_path.write_text(json.dumps({"engine": "dice"}), encoding="utf-8")
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()

    def fake_run_subprocess(**kwargs):
        output_root = Path(kwargs["command"][-1])
        run_dir = output_root / "20260429_000000_dice"
        run_dir.mkdir()
        _write_jsonl(
            run_dir / "cases.jsonl",
            [
                {
                    "status": "FEASIBLE",
                    "reason_code": "OK",
                    "runtime_ms": 12.0,
                    "candidate_count": 0,
                }
            ],
        )
        (run_dir / "candidates.csv").write_text("", encoding="utf-8")
        return {
            "status": "completed",
            "returncode": 0,
            "timeout_seconds": kwargs["timeout_seconds"],
            "runtime_seconds": 0.1,
            "stdout": "ok",
            "stderr": "",
        }

    monkeypatch.setattr(
        "experiments.scripts.run_baseline._run_subprocess",
        fake_run_subprocess,
    )

    scenario_root, steps = run_baseline_scenarios(
        config_paths=[config_path],
        engine_config_path=engine_config_path,
        baseline_root=baseline_root,
        scenario_limit=3,
        scenario_timeout_seconds=5,
    )

    rows = _read_csv(scenario_root / "scenario_summary.csv")
    effective_config = json.loads(
        (scenario_root / "dice_case" / "effective_config.json").read_text(encoding="utf-8")
    )

    assert effective_config["limit"] == 3
    assert steps[0]["status"] == "completed"
    assert rows[0]["scenario"] == "dice_case"
    assert rows[0]["target_success_rate_all_candidates"] == "0.0"
    assert rows[0]["mean_runtime_ms"] == "12.0"


def test_run_baseline_scenarios_records_timeout_without_stopping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "dice_slow.json"
    config_path.write_text(json.dumps({"engine": "dice"}), encoding="utf-8")
    engine_config_path = tmp_path / "engine.json"
    engine_config_path.write_text(json.dumps({"engine": "dice"}), encoding="utf-8")
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()

    def fake_run_subprocess(**kwargs):
        return {
            "status": "timeout",
            "returncode": None,
            "timeout_seconds": kwargs["timeout_seconds"],
            "runtime_seconds": 5.0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(
        "experiments.scripts.run_baseline._run_subprocess",
        fake_run_subprocess,
    )

    scenario_root, steps = run_baseline_scenarios(
        config_paths=[config_path],
        engine_config_path=engine_config_path,
        baseline_root=baseline_root,
        scenario_limit=None,
        scenario_timeout_seconds=5,
    )

    rows = _read_csv(scenario_root / "scenario_summary.csv")
    step_results = json.loads(
        (scenario_root / "scenario_step_results.json").read_text(encoding="utf-8")
    )

    assert steps[0]["status"] == "timeout"
    assert step_results["steps"][0]["status"] == "timeout"
    assert rows == []
