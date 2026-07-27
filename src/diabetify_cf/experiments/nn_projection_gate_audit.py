from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diabetify_cf.engine.nn_engine import NearestNeighborCounterfactualEngine
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblation,
    NNProjectionAblationConfig,
)
from diabetify_cf.schemas import CounterfactualRequest, JSONFeatureValue
from diabetify_cf.verification.fixtures import load_verification_scenarios


@dataclass(frozen=True)
class GateAuditCounts:
    raw_generated_candidates: int
    duplicate_candidates_removed: int
    total_candidates: int
    failed_directional: int
    failed_transition: int
    failed_medical: int
    failed_target: int
    valid_remaining: int

    def to_payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileGateAudit:
    profile_id: str
    reference_index: int | None
    baseline_probability_low_risk: float
    counts: GateAuditCounts

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "profile_id": self.profile_id,
            "baseline_probability_low_risk": self.baseline_probability_low_risk,
            "counts": self.counts.to_payload(),
        }
        if self.reference_index is not None:
            payload["reference_index"] = self.reference_index
        return payload


class NNProjectionGateAudit:
    """Audit sequential production gates over all projected NN candidates."""

    def __init__(
        self,
        *,
        engine: NearestNeighborCounterfactualEngine,
        config: NNProjectionAblationConfig,
        profiles_path: str | Path,
    ) -> None:
        if engine.artifacts is None:
            detail = engine.initialization_error or "unknown artifact initialization error"
            raise RuntimeError(f"counterfactual engine is not ready: {detail}")

        self.engine = engine
        self.config = config.validated()
        self.profiles_path = Path(profiles_path)
        self.ablation = NNProjectionAblation(engine=engine, config=config)
        self.artifacts = engine.artifacts

    def run(self) -> dict[str, Any]:
        profiles = self._load_profiles()
        audits = [self.audit_profile(profile) for profile in profiles]
        aggregate = _sum_counts([audit.counts for audit in audits])
        return {
            "experiment": "nn_projection_sequential_gate_audit",
            "description": (
                "Sequential gate audit over unique prefix-sparse NN projection "
                "candidates generated for each selected profile."
            ),
            "config": self.config.fixed_profile_payload(),
            "profiles_source": str(self.profiles_path),
            "candidate_generation": {
                "neighbor_count": self.config.max_neighbors,
                "projection_variants": ["prefix_sparse"],
                "deduplication": "feature-state deduplication before gate audit",
            },
            "gate_order": [
                "directional_constraint",
                "allowed_transition",
                "medical_range",
                "target_probability",
            ],
            "summary": _counts_summary(aggregate),
            "profiles": [audit.to_payload() for audit in audits],
        }

    def audit_profile(self, profile: dict[str, Any]) -> ProfileGateAudit:
        baseline = {str(feature): value for feature, value in dict(profile["features"]).items()}
        mutable_allowed = list(profile.get("mutable_allowed", self.config.mutable_allowed))
        request = CounterfactualRequest.model_validate(
            {
                "request_id": f"gate-audit-{profile['profile_id']}",
                "target": {
                    "target_class": self.config.target_class,
                    "min_target_probability": self.config.min_target_probability,
                },
                "instance": {"features": baseline},
                "constraints": {"mutable_allowed": mutable_allowed},
                "generation": {"timeout_ms": 60000},
            }
        )
        prepared = self.engine._prepare_request(request)
        ranked_neighbors = self.engine._rank_neighbors(
            eligible=self.ablation.eligible_frame,
            eligible_probabilities=self.ablation.eligible_probabilities,
            baseline=prepared.instance_features,
            mutable_allowed=prepared.mutable_allowed,
            prepared=prepared,
        )[: self.config.candidate_pool_size]
        ranked_neighbors = ranked_neighbors[: self.config.max_neighbors]
        specs = self.ablation._build_projection_specs(
            ranked_neighbors=ranked_neighbors,
            prepared=prepared,
        )
        specs = [spec for spec in specs if spec.method == "sparse"]
        candidates = self._unique_candidate_features(
            [spec.features for spec in specs],
            prepared.model_columns,
        )
        counts = self._audit_candidates(
            raw_generated_candidates=len(specs),
            candidates=candidates,
            request=request,
            prepared=prepared,
        )
        return ProfileGateAudit(
            profile_id=str(profile["profile_id"]),
            reference_index=(
                int(profile["reference_index"])
                if profile.get("reference_index") is not None
                else None
            ),
            baseline_probability_low_risk=prepared.base_prediction.probability_low_risk,
            counts=counts,
        )

    def _audit_candidates(
        self,
        *,
        raw_generated_candidates: int,
        candidates: list[dict[str, JSONFeatureValue]],
        request: CounterfactualRequest,
        prepared: Any,
    ) -> GateAuditCounts:
        total_candidates = len(candidates)
        mutable_set = set(prepared.mutable_allowed)
        directional_passed = [
            candidate
            for candidate in candidates
            if self.engine._directional_ok(
                candidate=candidate,
                baseline=prepared.instance_features,
                mutable_allowed=mutable_set,
                registry=prepared.registry,
            )
        ]
        transition_passed = [
            candidate
            for candidate in directional_passed
            if self.engine._transition_ok(
                candidate=candidate,
                baseline=prepared.instance_features,
                mutable_allowed=mutable_set,
                registry=prepared.registry,
            )
        ]
        medical_passed = [
            candidate
            for candidate in transition_passed
            if self.engine._medical_ok(candidate, prepared.registry)
        ]
        target_passed = self._target_passed(
            candidates=medical_passed,
            request=request,
            model_columns=prepared.model_columns,
        )
        return GateAuditCounts(
            raw_generated_candidates=raw_generated_candidates,
            duplicate_candidates_removed=raw_generated_candidates - total_candidates,
            total_candidates=total_candidates,
            failed_directional=total_candidates - len(directional_passed),
            failed_transition=len(directional_passed) - len(transition_passed),
            failed_medical=len(transition_passed) - len(medical_passed),
            failed_target=len(medical_passed) - len(target_passed),
            valid_remaining=len(target_passed),
        )

    def _target_passed(
        self,
        *,
        candidates: list[dict[str, JSONFeatureValue]],
        request: CounterfactualRequest,
        model_columns: list[str],
    ) -> list[dict[str, JSONFeatureValue]]:
        if not candidates:
            return []
        frame = pd.DataFrame(candidates, columns=model_columns)
        frame = self.engine._as_model_input_df(frame)
        probabilities = np.asarray(self.artifacts.model.predict_proba(frame)[:, 0], dtype=float)
        return [
            candidate
            for candidate, probability in zip(candidates, probabilities, strict=True)
            if probability >= request.target.min_target_probability
        ]

    def _unique_candidate_features(
        self,
        candidates: list[dict[str, JSONFeatureValue]],
        model_columns: list[str],
    ) -> list[dict[str, JSONFeatureValue]]:
        unique: list[dict[str, JSONFeatureValue]] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate in candidates:
            key = tuple(_stable_feature_value(candidate[column]) for column in model_columns)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _load_profiles(self) -> list[dict[str, Any]]:
        payload = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        if "scenarios" in payload:
            return [
                {
                    "profile_id": scenario.name,
                    "reference_index": None,
                    "features": dict(scenario.request.instance.features),
                    "mutable_allowed": list(scenario.request.constraints.mutable_allowed),
                }
                for scenario in load_verification_scenarios(self.profiles_path)
            ]
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list) or not profiles:
            raise ValueError(f"no profiles found in {self.profiles_path}")
        return [dict(profile) for profile in profiles]


def _stable_feature_value(value: JSONFeatureValue) -> JSONFeatureValue:
    if isinstance(value, float):
        return round(value, 9)
    return value


def _sum_counts(counts: list[GateAuditCounts]) -> GateAuditCounts:
    return GateAuditCounts(
        raw_generated_candidates=sum(item.raw_generated_candidates for item in counts),
        duplicate_candidates_removed=sum(item.duplicate_candidates_removed for item in counts),
        total_candidates=sum(item.total_candidates for item in counts),
        failed_directional=sum(item.failed_directional for item in counts),
        failed_transition=sum(item.failed_transition for item in counts),
        failed_medical=sum(item.failed_medical for item in counts),
        failed_target=sum(item.failed_target for item in counts),
        valid_remaining=sum(item.valid_remaining for item in counts),
    )


def _counts_summary(counts: GateAuditCounts) -> dict[str, Any]:
    total = max(counts.total_candidates, 1)
    return {
        **counts.to_payload(),
        "failure_rates": {
            "directional": counts.failed_directional / total,
            "transition": counts.failed_transition / total,
            "medical": counts.failed_medical / total,
            "target": counts.failed_target / total,
        },
        "valid_rate": counts.valid_remaining / total,
    }
