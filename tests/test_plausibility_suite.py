from __future__ import annotations

from pathlib import Path

from diabetify_cf.config import Settings
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)
from diabetify_cf.verification import ExternalCounterfactualVerifier, ScenarioRunner
from diabetify_cf.verification.suites import (
    VerificationSuite,
    build_suite_payload,
    load_suite_scenarios,
)


def test_plausibility_core_reports_lof_without_gating_generation() -> None:
    settings = Settings()
    suite = VerificationSuite(
        name="plausibility_core",
        description=(
            "Diverse full-actionable user profiles used to evaluate LOF-based plausibility."
        ),
        include_tags=("plausibility",),
    )
    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)
    engine = NearestNeighborCounterfactualEngine(
        model_path=settings.model_path,
        columns_path=settings.columns_path,
        reference_data_path=settings.reference_data_path,
        feature_registry_path=settings.feature_registry_path,
        options=NearestNeighborOptions.from_settings(settings),
    )
    verifier = ExternalCounterfactualVerifier(settings=settings)
    runner = ScenarioRunner(engine=engine, verifier=verifier)

    aggregates = runner.run(scenarios)
    summary = runner.summarize(aggregates)
    payload = build_suite_payload(suite=suite, aggregates=aggregates, summary=summary)

    assert [scenario.name for scenario in scenarios] == [
        "plausibility_full_actionable_profile_01",
        "plausibility_full_actionable_profile_02",
        "plausibility_full_actionable_profile_03",
        "plausibility_full_actionable_profile_04",
        "plausibility_full_actionable_profile_05",
        "plausibility_full_actionable_profile_06",
        "plausibility_full_actionable_profile_07",
        "plausibility_full_actionable_profile_08",
        "plausibility_full_actionable_profile_09",
        "plausibility_full_actionable_profile_10",
        "plausibility_full_actionable_profile_11",
        "plausibility_full_actionable_profile_12",
        "plausibility_full_actionable_profile_13",
        "plausibility_full_actionable_profile_14",
        "plausibility_full_actionable_profile_15",
        "plausibility_full_actionable_profile_16",
        "plausibility_full_actionable_profile_17",
        "plausibility_full_actionable_profile_18",
        "plausibility_full_actionable_profile_19",
        "plausibility_smoker_full_actionable_profile_20",
        "plausibility_smoker_full_actionable_profile_21",
        "plausibility_smoker_full_actionable_profile_22",
        "plausibility_smoker_full_actionable_profile_23",
        "plausibility_smoker_full_actionable_profile_24",
        "plausibility_smoker_full_actionable_profile_25",
    ]
    assert summary.total_scenarios == 25
    assert summary.total_candidates == 25
    assert summary.average_lof_score is not None
    assert summary.min_lof_score is not None
    assert summary.maximum_lof_score is not None
    assert summary.min_lof_score <= summary.average_lof_score <= summary.maximum_lof_score
    assert payload["summary"]["min_lof_score"] == summary.min_lof_score
    assert payload["summary"]["maximum_lof_score"] == summary.maximum_lof_score
    assert all(item["lof_score"] is not None for item in payload["scenarios"])
