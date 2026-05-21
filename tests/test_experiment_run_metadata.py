from pathlib import Path

from experiments.scripts.run_metadata import TRACKED_PACKAGES, build_run_metadata


def test_build_run_metadata_contains_reproducibility_fields() -> None:
    metadata = build_run_metadata(
        repo_root=Path.cwd(),
        run_type="benchmark",
        engine_name="dice",
        config_path=Path("experiments/configs/scenarios/all_mutable.json"),
    )

    assert metadata["run_type"] == "benchmark"
    assert metadata["engine_name"] == "dice"
    assert metadata["config_path"] == "experiments/configs/scenarios/all_mutable.json"
    assert "version" in metadata["python"]
    assert "system" in metadata["platform"]
    assert "commit" in metadata["git"]
    assert set(TRACKED_PACKAGES).issubset(metadata["packages"])
