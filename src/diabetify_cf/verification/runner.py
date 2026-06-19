from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any
from typing import TypeVar

from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest, CounterfactualResponse
from diabetify_cf.verification.external import ExternalCounterfactualVerifier, VerificationReport


@dataclass(frozen=True)
class ScenarioExpectation:
    expected_status: Status
    expected_reason_codes: tuple[ReasonCode, ...] = ()
    no_solution_expected: bool = False

    def matches(self, response: CounterfactualResponse) -> bool:
        if response.status != self.expected_status:
            return False
        if self.expected_reason_codes and response.reason_code not in self.expected_reason_codes:
            return False
        if self.no_solution_expected:
            return len(response.candidates) == 0
        return True


@dataclass(frozen=True)
class VerificationScenario:
    name: str
    request: CounterfactualRequest
    expectation: ScenarioExpectation
    repeat_count: int = 1
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be at least 1")


@dataclass(frozen=True)
class ScenarioRunRecord:
    scenario_name: str
    iteration: int
    response: CounterfactualResponse
    verification: VerificationReport
    duration_ms: int
    expectation_matched: bool

    @property
    def passed(self) -> bool:
        return self.expectation_matched and self.verification.outcome_consistent


@dataclass(frozen=True)
class ScenarioAggregate:
    scenario: VerificationScenario
    runs: list[ScenarioRunRecord]
    repeatability_consistent: bool

    @property
    def passed(self) -> bool:
        return all(run.passed for run in self.runs)

    @property
    def average_duration_ms(self) -> float:
        return mean(run.duration_ms for run in self.runs)


@dataclass(frozen=True)
class MetricSummary:
    immutable_violation_rate: float | None
    mutable_violation_rate: float | None
    repeatability_rate: float | None
    average_latency_ms: float
    p95_latency_ms: float
    total_scenarios: int
    total_runs: int
    total_candidates: int


class ScenarioRunner:
    def __init__(
        self,
        *,
        engine: CounterfactualEngine,
        verifier: ExternalCounterfactualVerifier,
    ) -> None:
        self.engine = engine
        self.verifier = verifier

    def run(self, scenarios: list[VerificationScenario]) -> list[ScenarioAggregate]:
        aggregates: list[ScenarioAggregate] = []
        for scenario in scenarios:
            runs: list[ScenarioRunRecord] = []
            for iteration in range(1, scenario.repeat_count + 1):
                started = perf_counter()
                response = self.engine.generate(scenario.request)
                duration_ms = int((perf_counter() - started) * 1000)
                verification = self.verifier.verify_response(
                    request=scenario.request,
                    response=response,
                )
                runs.append(
                    ScenarioRunRecord(
                        scenario_name=scenario.name,
                        iteration=iteration,
                        response=response,
                        verification=verification,
                        duration_ms=duration_ms,
                        expectation_matched=scenario.expectation.matches(response),
                    )
                )

            repeatability_consistent = _is_repeatability_consistent(runs)
            aggregates.append(
                ScenarioAggregate(
                    scenario=scenario,
                    runs=runs,
                    repeatability_consistent=repeatability_consistent,
                )
            )
        return aggregates

    def summarize(self, aggregates: list[ScenarioAggregate]) -> MetricSummary:
        candidate_results = [
            candidate_result
            for aggregate in aggregates
            for run in aggregate.runs
            for candidate_result in run.verification.candidate_results
        ]
        all_runs = [run for aggregate in aggregates for run in aggregate.runs]
        repeatability_scenarios = [
            aggregate for aggregate in aggregates if aggregate.scenario.repeat_count > 1
        ]
        latencies = [run.duration_ms for run in all_runs]

        return MetricSummary(
            immutable_violation_rate=_ratio(
                candidate_results,
                lambda item: not item.immutable_ok,
            ),
            mutable_violation_rate=_ratio(
                candidate_results,
                lambda item: not item.mutable_ok,
            ),
            repeatability_rate=_ratio(
                repeatability_scenarios,
                lambda aggregate: aggregate.repeatability_consistent,
            ),
            average_latency_ms=mean(latencies) if latencies else 0.0,
            p95_latency_ms=_percentile(latencies, 95.0),
            total_scenarios=len(aggregates),
            total_runs=len(all_runs),
            total_candidates=len(candidate_results),
        )


def _is_repeatability_consistent(runs: list[ScenarioRunRecord]) -> bool:
    if len(runs) <= 1:
        return True

    baseline = _response_signature(runs[0].response)
    return all(_response_signature(run.response) == baseline for run in runs[1:])


def _response_signature(response: CounterfactualResponse) -> tuple[Any, ...]:
    candidate_signatures = tuple(
        (
            candidate.candidate_id,
            _normalized_feature_items(candidate.features),
            tuple(sorted((name, round(float(delta), 6)) for name, delta in candidate.delta.items())),
            candidate.prediction.class_name,
            round(candidate.prediction.probability_low_risk, 9),
        )
        for candidate in response.candidates
    )
    return (
        response.status,
        response.reason_code,
        candidate_signatures,
    )


def _normalized_feature_items(features: dict[str, object]) -> tuple[tuple[str, object], ...]:
    normalized: list[tuple[str, object]] = []
    for key, value in sorted(features.items()):
        if isinstance(value, bool):
            normalized.append((key, value))
        elif isinstance(value, (int, float)):
            normalized.append((key, round(float(value), 6)))
        else:
            normalized.append((key, value))
    return tuple(normalized)


TItem = TypeVar("TItem")


def _ratio(items: list[TItem], predicate: Callable[[TItem], bool]) -> float | None:
    if not items:
        return None
    return sum(1 for item in items if predicate(item)) / len(items)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
