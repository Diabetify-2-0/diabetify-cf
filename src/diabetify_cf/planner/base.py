from __future__ import annotations

from abc import ABC, abstractmethod

from diabetify_cf.schemas import (
    CounterfactualCandidate,
    CounterfactualRequest,
    PlannerInput,
    PrescriptivePlan,
)


class PrescriptivePlanner(ABC):
    @abstractmethod
    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
        planner_input: PlannerInput,
    ) -> PrescriptivePlan:
        raise NotImplementedError


class FallbackPrescriptivePlanner(PrescriptivePlanner):
    def __init__(self, primary: PrescriptivePlanner, fallback: PrescriptivePlanner) -> None:
        self.primary = primary
        self.fallback = fallback

    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
        planner_input: PlannerInput,
    ) -> PrescriptivePlan:
        try:
            return self.primary.build_plan(
                request=request,
                candidate=candidate,
                planner_input=planner_input,
            )
        except Exception:
            return self.fallback.build_plan(
                request=request,
                candidate=candidate,
                planner_input=planner_input,
            )
