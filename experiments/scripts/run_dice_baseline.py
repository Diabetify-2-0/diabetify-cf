from __future__ import annotations

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError: 
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.run_baseline import ( 
    DEFAULT_SCENARIO_TIMEOUT_SECONDS,
    DEFAULT_STABILITY_CONFIG,
    DEFAULT_STABILITY_TIMEOUT_SECONDS,
    apply_limit,
    main,
    run_baseline_scenarios,
    run_baseline_stability,
    run_engine_baseline,
)

run_dice_baseline = run_engine_baseline

__all__ = [
    "DEFAULT_SCENARIO_TIMEOUT_SECONDS",
    "DEFAULT_STABILITY_CONFIG",
    "DEFAULT_STABILITY_TIMEOUT_SECONDS",
    "apply_limit",
    "run_baseline_scenarios",
    "run_baseline_stability",
    "run_dice_baseline",
    "run_engine_baseline",
]


if __name__ == "__main__":
    main()
