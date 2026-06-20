from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from diabetify_cf.config import Settings
from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.engine.shared import ArtifactBackedCounterfactualEngine, PreparedRequest
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CounterfactualCandidate,
    CounterfactualRequest,
    CounterfactualResponse,
    JSONFeatureValue,
    PredictionInfo,
)


@dataclass(frozen=True)
class VerificationCandidateResult:
    candidate_id: str
    immutable_ok: bool
    mutable_ok: bool
    target_satisfied: bool
    prediction_matches_returned: bool
    external_lof_score: float
    external_lof_within_threshold: bool
    medical_ok: bool
    directional_ok: bool
    transition_ok: bool

    @property
    def constraint_gate_ok(self) -> bool:
        return self.medical_ok and self.directional_ok and self.transition_ok


@dataclass(frozen=True)
class VerificationReport:
    request_id: str
    response_status: Status
    candidate_results: list[VerificationCandidateResult]
    outcome_consistent: bool

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_results)

    @property
    def immutable_violation_rate(self) -> float | None:
        return _rate(self.candidate_results, lambda item: not item.immutable_ok)

    @property
    def mutable_violation_rate(self) -> float | None:
        return _rate(self.candidate_results, lambda item: not item.mutable_ok)


class _VerifierRuleChecker(ArtifactBackedCounterfactualEngine):
    def __init__(self, *, artifacts: ModelArtifacts, max_lof_score: float) -> None:
        self.max_lof_score = max(1.0, float(max_lof_score))
        self.logger = logging.getLogger("diabetify_cf.verification")
        self.artifacts = artifacts
        self.initialization_error = None

    def _generate_raw_candidates(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
    ) -> pd.DataFrame:
        raise NotImplementedError("Verification helper does not generate candidates.")


class ExternalCounterfactualVerifier:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        artifacts: ModelArtifacts | None = None,
        max_lof_score: float | None = None,
    ) -> None:
        if artifacts is None:
            loaded_settings = settings or Settings()
            artifacts = load_artifacts(
                model_path=loaded_settings.model_path,
                columns_path=loaded_settings.columns_path,
                reference_data_path=loaded_settings.reference_data_path,
                feature_registry_path=loaded_settings.feature_registry_path,
                artifact_manifest_path=loaded_settings.artifact_manifest_path,
            )
            if max_lof_score is None:
                max_lof_score = loaded_settings.max_lof_score

        if max_lof_score is None:
            max_lof_score = 1.5

        self._checker = _VerifierRuleChecker(
            artifacts=artifacts,
            max_lof_score=max_lof_score,
        )

    def verify_response(
        self,
        *,
        request: CounterfactualRequest,
        response: CounterfactualResponse,
    ) -> VerificationReport:
        prepared = self._checker._prepare_request(request)
        baseline_target_satisfied = self._checker._target_satisfied(
            prediction=prepared.base_prediction,
            target_class=request.target.target_class,
            min_target_probability=request.target.min_target_probability,
        )
        candidate_results = [
            self._verify_candidate(
                request=request,
                prepared=prepared,
                candidate=candidate,
            )
            for candidate in response.candidates
        ]

        outcome_consistent = self._outcome_consistent(
            response_status=response.status,
            response_reason_code=response.reason_code,
            candidate_results=candidate_results,
            baseline_target_satisfied=baseline_target_satisfied,
        )
        return VerificationReport(
            request_id=response.request_id,
            response_status=response.status,
            candidate_results=candidate_results,
            outcome_consistent=outcome_consistent,
        )

    def _verify_candidate(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
        candidate: CounterfactualCandidate,
    ) -> VerificationCandidateResult:
        candidate_features = prepared.registry.canonicalize_feature_map(candidate.features)
        typed_candidate = self._checker._to_series(
            prepared.model_columns,
            candidate_features,
            prepared.registry,
        )
        candidate_df = pd.DataFrame([typed_candidate], columns=prepared.model_columns)
        candidate_df = self._checker._as_model_input_df(candidate_df)

        verified_prediction = self._checker._predict_info(candidate_df)
        immutable_set = _verification_immutable_set(
            request=request,
            default_immutable_features=prepared.registry.immutable_defaults(),
        )
        mutable_set = _verification_mutable_set(request=request)
        immutable_ok = _immutable_ok(
            candidate_features,
            prepared.instance_features,
            immutable_set,
        )
        mutable_ok = _mutable_ok(
            candidate_features,
            prepared.instance_features,
            mutable_set,
        )
        medical_ok = self._checker._medical_ok(candidate_features, prepared.registry)
        directional_ok = self._checker._directional_ok(
            candidate=candidate_features,
            baseline=prepared.instance_features,
            mutable_allowed=set(prepared.mutable_allowed),
            registry=prepared.registry,
        )
        transition_ok = self._checker._transition_ok(
            candidate=candidate_features,
            baseline=prepared.instance_features,
            mutable_allowed=set(prepared.mutable_allowed),
            registry=prepared.registry,
        )
        target_satisfied = self._checker._target_satisfied(
            prediction=verified_prediction,
            target_class=request.target.target_class,
            min_target_probability=request.target.min_target_probability,
        )
        external_lof_score = self._checker._lof_score(candidate_df)

        return VerificationCandidateResult(
            candidate_id=candidate.candidate_id,
            immutable_ok=immutable_ok,
            mutable_ok=mutable_ok,
            target_satisfied=target_satisfied,
            prediction_matches_returned=_prediction_matches(
                returned=candidate.prediction,
                verified=verified_prediction,
            ),
            external_lof_score=external_lof_score,
            external_lof_within_threshold=external_lof_score <= self._checker.max_lof_score,
            medical_ok=medical_ok,
            directional_ok=directional_ok,
            transition_ok=transition_ok,
        )

    @staticmethod
    def _outcome_consistent(
        *,
        response_status: Status,
        response_reason_code: ReasonCode,
        candidate_results: list[VerificationCandidateResult],
        baseline_target_satisfied: bool,
    ) -> bool:
        if response_status == Status.FEASIBLE:
            if not candidate_results:
                return (
                    response_reason_code == ReasonCode.TARGET_ALREADY_SATISFIED
                    and baseline_target_satisfied
                )
            return all(
                item.immutable_ok
                and item.mutable_ok
                and item.target_satisfied
                and item.external_lof_within_threshold
                and item.constraint_gate_ok
                for item in candidate_results
            )

        return len(candidate_results) == 0


def _prediction_matches(
    *,
    returned: PredictionInfo,
    verified: PredictionInfo,
    tolerance: float = 1e-9,
) -> bool:
    return (
        returned.class_name == verified.class_name
        and abs(returned.probability_low_risk - verified.probability_low_risk) <= tolerance
    )


def _rate(
    items: list[VerificationCandidateResult],
    predicate: Callable[[VerificationCandidateResult], bool],
) -> float | None:
    if not items:
        return None
    matched = sum(1 for item in items if predicate(item))
    return matched / len(items)


def _verification_immutable_set(
    *,
    request: CounterfactualRequest,
    default_immutable_features: list[str],
) -> set[str]:
    return set(default_immutable_features).union(request.constraints.immutable_features)


def _verification_mutable_set(*, request: CounterfactualRequest) -> set[str]:
    return set(request.constraints.mutable_allowed)


def _immutable_ok(
    candidate: dict[str, JSONFeatureValue],
    baseline: dict[str, JSONFeatureValue],
    immutable_set: set[str],
) -> bool:
    for feature in immutable_set:
        if feature not in candidate or feature not in baseline:
            continue
        if not _equal(candidate[feature], baseline[feature]):
            return False
    return True


def _mutable_ok(
    candidate: dict[str, JSONFeatureValue],
    baseline: dict[str, JSONFeatureValue],
    mutable_set: set[str],
) -> bool:
    for feature, value in candidate.items():
        if feature not in baseline:
            continue
        if not _equal(value, baseline[feature]) and feature not in mutable_set:
            return False
    return True


def _equal(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b
