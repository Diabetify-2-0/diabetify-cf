from __future__ import annotations

from abc import ABC, abstractmethod

from diabetify_cf.schemas import CounterfactualRequest, CounterfactualResponse


class CounterfactualEngine(ABC):
    @abstractmethod
    def generate(self, request: CounterfactualRequest) -> CounterfactualResponse:
        raise NotImplementedError
