from __future__ import annotations

from pathlib import Path

from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.verification import load_verification_scenarios


def test_load_verification_scenarios_from_directory() -> None:
    scenarios = load_verification_scenarios(Path("evaluation") / "fixtures")

    assert len(scenarios) >= 6
    feasible = next(item for item in scenarios if item.name == "feasible_bmi_activity")
    repeatable = next(
        item for item in scenarios if item.name == "feasible_bmi_activity_repeatability"
    )
    already_satisfied = next(
        item for item in scenarios if item.name == "feasible_target_already_satisfied"
    )
    infeasible = next(item for item in scenarios if item.name == "infeasible_no_mutable")
    target_unreachable = next(
        item for item in scenarios if item.name == "infeasible_target_unreachable_bmi_only"
    )
    medical_only = next(
        item for item in scenarios if item.name == "infeasible_medical_rule_only_high_target"
    )

    assert feasible.repeat_count == 1
    assert feasible.tags == ("feasible", "service", "production")
    assert feasible.expectation.expected_status == Status.FEASIBLE
    assert feasible.expectation.expected_reason_codes == (ReasonCode.OK,)

    assert repeatable.repeat_count == 3
    assert repeatable.tags == ("feasible", "repeatability", "service", "production")
    assert repeatable.expectation.expected_status == Status.FEASIBLE
    assert repeatable.expectation.expected_reason_codes == (ReasonCode.OK,)

    assert already_satisfied.expectation.expected_status == Status.FEASIBLE
    assert already_satisfied.expectation.expected_reason_codes == (
        ReasonCode.TARGET_ALREADY_SATISFIED,
    )

    assert infeasible.expectation.expected_status == Status.INFEASIBLE
    assert infeasible.expectation.expected_reason_codes == (ReasonCode.NO_MUTABLE_FEATURE,)
    assert infeasible.expectation.no_solution_expected is True

    assert target_unreachable.expectation.expected_status == Status.INFEASIBLE
    assert target_unreachable.expectation.expected_reason_codes == (
        ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,
    )
    assert target_unreachable.expectation.no_solution_expected is True

    assert medical_only.expectation.expected_status == Status.INFEASIBLE
    assert medical_only.expectation.expected_reason_codes == (
        ReasonCode.MEDICAL_RULE_VIOLATION_ONLY,
    )
    assert medical_only.expectation.no_solution_expected is True


def test_load_verification_scenarios_can_filter_by_tags() -> None:
    scenarios = load_verification_scenarios(
        Path("evaluation") / "fixtures",
        include_tags=("repeatability",),
    )

    assert [item.name for item in scenarios] == ["feasible_bmi_activity_repeatability"]


def test_load_verification_scenarios_can_exclude_by_tags() -> None:
    scenarios = load_verification_scenarios(
        Path("evaluation") / "fixtures",
        exclude_tags=("repeatability",),
    )

    names = {item.name for item in scenarios}

    assert "feasible_bmi_activity_repeatability" not in names
    assert "feasible_bmi_activity" in names
    assert "infeasible_no_mutable" in names
