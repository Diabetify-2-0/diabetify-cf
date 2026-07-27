from __future__ import annotations

from pathlib import Path

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.verification import load_verification_scenarios


def test_load_verification_scenarios_from_directory() -> None:
    scenarios = load_verification_scenarios(Path("evaluation") / "fixtures")

    assert len(scenarios) >= 6
    feasible = next(item for item in scenarios if item.name == "actionability_non_bmi_only")
    repeatable = next(item for item in scenarios if item.name == "consistency_profile_01")
    medical_infeasible_repeatable = next(
        item for item in scenarios if item.name == "consistency_profile_12"
    )
    plausibility = next(
        item for item in scenarios if item.name == "plausibility_full_actionable_profile_01"
    )

    assert feasible.repeat_count == 1
    assert feasible.tags == (
        "actionability_profile",
        "actionability",
        "service",
        "production",
        "feasible",
    )
    assert feasible.expectation.expected_status == Status.FEASIBLE
    assert feasible.expectation.expected_reason_codes == (ReasonCode.OK,)

    assert repeatable.repeat_count == 10
    assert repeatable.tags == (
        "consistency_profile",
        "feasible",
        "no_smoking_change",
        "service",
        "production",
    )
    assert repeatable.expectation.expected_status == Status.FEASIBLE
    assert repeatable.expectation.expected_reason_codes == (ReasonCode.OK,)

    assert medical_infeasible_repeatable.expectation.expected_status == Status.INFEASIBLE
    assert medical_infeasible_repeatable.expectation.expected_reason_codes == (
        ReasonCode.MEDICAL_RULE_VIOLATION_ONLY,
    )
    assert medical_infeasible_repeatable.expectation.no_solution_expected is True

    assert plausibility.expectation.expected_status == Status.FEASIBLE
    assert plausibility.expectation.expected_reason_codes == (ReasonCode.OK,)
    assert plausibility.tags == (
        "plausibility",
        "feasible",
        "full_actionable_no_smoking_change",
        "service",
        "production",
    )


def test_load_verification_scenarios_can_filter_by_tags() -> None:
    scenarios = load_verification_scenarios(
        Path("evaluation") / "fixtures",
        include_tags=("consistency_profile",),
    )

    assert [item.name for item in scenarios] == [
        "consistency_profile_01",
        "consistency_profile_02",
        "consistency_profile_03",
        "consistency_profile_04",
        "consistency_profile_05",
        "consistency_profile_06",
        "consistency_profile_07",
        "consistency_profile_08",
        "consistency_profile_09",
        "consistency_profile_10",
        "consistency_profile_11",
        "consistency_profile_12",
    ]


def test_load_verification_scenarios_can_exclude_by_tags() -> None:
    scenarios = load_verification_scenarios(
        Path("evaluation") / "fixtures",
        exclude_tags=("consistency_profile",),
    )

    names = {item.name for item in scenarios}

    assert "consistency_profile_01" not in names
    assert "actionability_non_bmi_activity_hypertension" in names
