from pathlib import Path

from experiments.scripts.run_core_checkpoint import (
    build_checkpoint_command,
    core_engine_config_paths,
    core_engine_names,
    latest_comparison_pointer,
    load_scope,
)


def test_load_scope_reads_core_benchmark_definition() -> None:
    scope = load_scope(Path("experiments/configs/benchmark_scope/core_benchmark.json"))

    assert scope["name"] == "core_benchmark_v1"
    assert [entry["engine"] for entry in scope["core_engines"]] == [
        "dice",
        "ocean",
        "ft",
        "nn",
    ]


def test_core_engine_helpers_preserve_locked_engine_order() -> None:
    scope = load_scope(Path("experiments/configs/benchmark_scope/core_benchmark.json"))

    config_names = [path.name for path in core_engine_config_paths(scope)]
    engine_names = core_engine_names(scope)

    assert config_names == ["dice.json", "ocean.json", "ft.json", "nn.json"]
    assert engine_names == ["dice", "ocean", "ft", "nn"]


def test_build_checkpoint_command_uses_core_scope_and_required_engines() -> None:
    scope = load_scope(Path("experiments/configs/benchmark_scope/core_benchmark.json"))

    command = build_checkpoint_command(
        scope=scope,
        output_root=Path("experiments/results"),
        scenario_limit=12,
        stability_limit=8,
        repeat_count=4,
        scenario_timeout_seconds=240,
        stability_timeout_seconds=300,
    )

    assert "--engine-configs" in command
    assert "--required-engines" in command
    assert "--fail-on-timeout" in command
    required_index = command.index("--required-engines")
    strict_index = command.index("--fail-on-timeout")
    assert command[required_index + 1 : strict_index] == ["dice", "ocean", "ft", "nn"]
    path_names = {Path(value).name for value in command if value.endswith(".json")}
    assert "dice.json" in path_names
    assert "nn.json" in path_names


def test_latest_comparison_pointer_uses_requested_output_root() -> None:
    pointer = latest_comparison_pointer(Path("custom_results_root"))

    assert pointer == Path("custom_results_root/latest/comparison.txt")
