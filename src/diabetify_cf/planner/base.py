from __future__ import annotations

from abc import ABC, abstractmethod

from diabetify_cf.schemas import CounterfactualCandidate, CounterfactualRequest, PrescriptivePlan


class PrescriptivePlanner(ABC):
    @abstractmethod
    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
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
    ) -> PrescriptivePlan:
        try:
            return self.primary.build_plan(request=request, candidate=candidate)
        except Exception:  
            return self.fallback.build_plan(request=request, candidate=candidate)
