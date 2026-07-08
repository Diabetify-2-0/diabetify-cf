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


def test_plausibility_core_runs_production_feasible_scenarios_within_lof_threshold() -> None:
    settings = Settings()
    suite = VerificationSuite(
        name="plausibility_core",
        description="Feasible non-repeatability scenarios used to evaluate LOF-based plausibility.",
        include_tags=("feasible",),
        exclude_tags=("repeatability",),
    )
    scenarios = load_suite_scenarios(Path("evaluation") / "fixtures", suite)
    engine = NearestNeighborCounterfactualEngine(
        model_path=settings.model_path,
        columns_path=settings.columns_path,
        reference_data_path=settings.reference_data_path,
        feature_registry_path=settings.feature_registry_path,
        max_lof_score=settings.max_lof_score,
        options=NearestNeighborOptions.from_settings(settings),
    )
    verifier = ExternalCounterfactualVerifier(settings=settings)
    runner = ScenarioRunner(engine=engine, verifier=verifier)

    aggregates = runner.run(scenarios)
    summary = runner.summarize(aggregates)
    payload = build_suite_payload(suite=suite, aggregates=aggregates, summary=summary)

    assert [scenario.name for scenario in scenarios] == [
        "feasible_bmi_activity",
        "feasible_bmi_cholesterol",
        "feasible_bmi_hypertension",
        "feasible_bmi_only",
        "feasible_full_actionable",
        "feasible_smoker_bmi_hypertension",
        "feasible_smoker_bmi_hypertension_activity",
        "feasible_smoker_full_actionable",
    ]
    assert all(aggregate.passed for aggregate in aggregates)
    assert summary.total_scenarios == 8
    assert summary.total_candidates == 8
    assert summary.lof_violation_rate == 0.0
    assert summary.average_lof_score is not None
    assert summary.average_lof_score <= settings.max_lof_score
    assert payload["summary"]["lof_within_threshold"] is True
    assert all(item["lof_score"] is not None for item in payload["scenarios"])
    assert all(item["lof_score"] <= settings.max_lof_score for item in payload["scenarios"])
