from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from diabetify_cf.engine.nn_engine import NearestNeighborCounterfactualEngine
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblation,
    NNProjectionAblationConfig,
)
from diabetify_cf.experiments.nn_projection_gate_audit import NNProjectionGateAudit
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
)
from diabetify_cf.verification.fixtures import load_verification_scenarios

OBJECTIVE_CONFIGS: dict[str, dict[str, float]] = {
    "proximity_only": {
        "proximity": 1.0,
        "plausibility": 0.0,
    },
    "plausibility_only": {
        "proximity": 0.0,
        "plausibility": 1.0,
    },
    "proximity_75_plausibility_25": {
        "proximity": 0.75,
        "plausibility": 0.25,
    },
    "proximity_50_plausibility_50": {
        "proximity": 0.50,
        "plausibility": 0.50,
    },
    "proximity_25_plausibility_75": {
        "proximity": 0.25,
        "plausibility": 0.75,
    },
}

OBJECTIVE_WEIGHT_SWEEP_CONFIGS = OBJECTIVE_CONFIGS


@dataclass(frozen=True)
class ValidProjectionCandidate:
    candidate: CounterfactualCandidate
    neighbor_rank: int
    projection_method: str
    prefix_length: int

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "neighbor_rank": self.neighbor_rank,
            "prefix_length": self.prefix_length,
            "target_probability": self.candidate.prediction.probability_low_risk,
            "proximity": self.candidate.metrics.distance_l1,
            "plausibility_lof": self.candidate.metrics.lof_score,
            "delta": self.candidate.delta,
            "features": self.candidate.features,
        }


@dataclass(frozen=True)
class ObjectiveSelectionProfile:
    profile_id: str
    reference_index: int | None
    request: CounterfactualRequest


class NNObjectiveSelectionSensitivity:
    """Compare best-candidate choices under different objective weight settings."""

    def __init__(
        self,
        *,
        engine: NearestNeighborCounterfactualEngine,
        config: NNProjectionAblationConfig,
        gate_audit_path: str | Path | None = None,
        scenario_paths: tuple[str | Path, ...] = (),
        objective_configs: dict[str, dict[str, float]] | None = None,
        experiment_name: str = "nn_objective_weight_sweep",
        description: str | None = None,
        include_diagnostics: bool = True,
    ) -> None:
        if engine.artifacts is None:
            detail = engine.initialization_error or "unknown artifact initialization error"
            raise RuntimeError(f"counterfactual engine is not ready: {detail}")

        self.engine = engine
        self.config = config.validated()
        self.gate_audit_path = Path(gate_audit_path) if gate_audit_path is not None else None
        self.scenario_paths = tuple(Path(path) for path in scenario_paths)
        self.objective_configs = (
            objective_configs if objective_configs is not None else OBJECTIVE_CONFIGS
        )
        self.experiment_name = experiment_name
        self.include_diagnostics = include_diagnostics
        self.description = description or (
            "Compare the best valid projection candidate selected by proximity-only, "
            "plausibility-only, and proximity-plausibility weighted objectives for profiles "
            "with at least one valid candidate."
        )
        if self.gate_audit_path is None and not self.scenario_paths:
            raise ValueError("gate_audit_path or scenario_paths must be provided")
        self.gate_audit_payload = (
            json.loads(self.gate_audit_path.read_text(encoding="utf-8"))
            if self.gate_audit_path is not None
            else None
        )
        self.profiles_path = (
            Path(self.gate_audit_payload["profiles_source"])
            if self.gate_audit_payload is not None
            else None
        )
        self.ablation = NNProjectionAblation(engine=engine, config=config)
        self.gate_audit = (
            NNProjectionGateAudit(
                engine=engine,
                config=config,
                profiles_path=self.profiles_path,
            )
            if self.profiles_path is not None
            else None
        )

    def run(self) -> dict[str, Any]:
        profiles = self._profiles()
        profile_payloads = []
        skipped_profiles = []
        selected_by_config: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.objective_configs
        }
        for profile in profiles:
            valid_candidates = self._valid_candidates_for_profile(profile)
            if not valid_candidates:
                skipped_profiles.append(
                    {
                        "profile_id": profile.profile_id,
                        "reference_index": profile.reference_index,
                        "reason": "no_valid_candidate_after_sequential_gates",
                    }
                )
                continue
            selections = self._select_by_objective_configs(
                valid_candidates,
                mutable_allowed=list(profile.request.constraints.mutable_allowed),
            )
            for config_name, selection in selections.items():
                selected_by_config[config_name].append(selection)
            profile_payloads.append(
                {
                    "profile_id": profile.profile_id,
                    "reference_index": profile.reference_index,
                    "valid_candidate_count": len(valid_candidates),
                    "selected_candidates": selections,
                }
            )

        payload = {
            "experiment": self.experiment_name,
            "description": self.description,
            "input_source": self._input_source_payload(),
            "profile_count": len(profile_payloads),
            "input_profile_count": len(profiles),
            "skipped_profile_count": len(skipped_profiles),
            "skipped_profiles": skipped_profiles,
            "objective_configs": self.objective_configs,
            "summary_by_objective": {
                config_name: _summary_for_selected(selected)
                for config_name, selected in selected_by_config.items()
            },
            "profiles": profile_payloads,
        }
        if self.include_diagnostics:
            payload["diagnostics"] = {
                "proximity_vs_plausibility_selection_overlap": (
                    _proximity_plausibility_overlap(profile_payloads)
                ),
            }
        return payload

    def _valid_candidates_for_profile(
        self,
        profile: ObjectiveSelectionProfile,
    ) -> list[ValidProjectionCandidate]:
        request = profile.request
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
        unique_specs = []
        seen: set[tuple[Any, ...]] = set()
        for spec in specs:
            key = self._state_key(spec.features, prepared.model_columns)
            if key in seen:
                continue
            seen.add(key)
            unique_specs.append(spec)

        valid_candidates: list[ValidProjectionCandidate] = []
        mutable_set = set(prepared.mutable_allowed)
        for index, spec in enumerate(unique_specs, start=1):
            candidate_features = spec.features
            if not self.engine._directional_ok(
                candidate=candidate_features,
                baseline=prepared.instance_features,
                mutable_allowed=mutable_set,
                registry=prepared.registry,
            ):
                continue
            if not self.engine._transition_ok(
                candidate=candidate_features,
                baseline=prepared.instance_features,
                mutable_allowed=mutable_set,
                registry=prepared.registry,
            ):
                continue
            if not self.engine._medical_ok(candidate_features, prepared.registry):
                continue

            candidate_df = pd.DataFrame([candidate_features], columns=prepared.model_columns)
            candidate_df = self.engine._as_model_input_df(candidate_df)
            prediction = self.engine._predict_info(candidate_df)
            if not self.engine._target_satisfied(
                prediction=prediction,
                target_class=request.target.target_class,
                min_target_probability=request.target.min_target_probability,
            ):
                continue

            delta = self.engine._build_delta(
                candidate_features,
                prepared.instance_features,
                mutable_set,
            )
            metrics = CandidateMetrics(
                distance_l1=self.engine._normalized_l1(
                    candidate_features,
                    prepared.instance_features,
                    prepared.registry,
                ),
                changed_feature_count=len(delta),
                lof_score=self.engine._lof_score(candidate_df),
            )
            candidate = CounterfactualCandidate(
                candidate_id=f"{profile.profile_id}_candidate_{index}",
                features=candidate_features,
                delta=delta,
                prediction=prediction,
                metrics=metrics,
            )
            valid_candidates.append(
                ValidProjectionCandidate(
                    candidate=candidate,
                    neighbor_rank=spec.neighbor_rank,
                    projection_method=spec.method,
                    prefix_length=spec.prefix_length,
                )
            )

        return valid_candidates

    def _select_by_objective_configs(
        self,
        candidates: list[ValidProjectionCandidate],
        mutable_allowed: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        selections: dict[str, dict[str, Any]] = {}
        registry = self.artifacts.feature_registry
        mutable_features = (
            mutable_allowed if mutable_allowed is not None else list(self.config.mutable_allowed)
        )
        objective_configs = getattr(self, "objective_configs", OBJECTIVE_CONFIGS)
        for config_name, weights in objective_configs.items():
            scored = [
                (
                    self.engine._objective_score(
                        candidate=item.candidate,
                        preferences=weights,
                        mutable_allowed=mutable_features,
                        registry=registry,
                    ),
                    item,
                )
                for item in candidates
            ]
            score, selected = min(
                scored,
                key=lambda item: (
                    item[0],
                    item[1].candidate.metrics.distance_l1,
                    item[1].candidate.metrics.changed_feature_count,
                    item[1].candidate.metrics.lof_score,
                    -item[1].candidate.prediction.probability_low_risk,
                ),
            )
            payload = selected.metrics_payload()
            payload["objective_score"] = score
            selections[config_name] = payload
        return selections

    @property
    def artifacts(self) -> Any:
        assert self.engine.artifacts is not None
        return self.engine.artifacts

    def _profiles(self) -> list[ObjectiveSelectionProfile]:
        if self.scenario_paths:
            return self._profiles_from_scenarios()
        return self._valid_profiles_from_gate_audit()

    @staticmethod
    def _state_key(
        candidate: dict[str, Any],
        model_columns: list[str],
    ) -> tuple[Any, ...]:
        key_parts: list[Any] = []
        for column in model_columns:
            value = candidate[column]
            if isinstance(value, float):
                key_parts.append(round(value, 9))
            else:
                key_parts.append(value)
        return tuple(key_parts)

    def _profiles_from_scenarios(self) -> list[ObjectiveSelectionProfile]:
        profiles: list[ObjectiveSelectionProfile] = []
        for path in self.scenario_paths:
            for scenario in load_verification_scenarios(path):
                profiles.append(
                    ObjectiveSelectionProfile(
                        profile_id=scenario.name,
                        reference_index=None,
                        request=scenario.request,
                    )
                )
        if not profiles:
            joined = ", ".join(str(path) for path in self.scenario_paths)
            raise ValueError(f"no scenarios found in {joined}")
        return profiles

    def _valid_profiles_from_gate_audit(self) -> list[ObjectiveSelectionProfile]:
        assert self.gate_audit_payload is not None
        assert self.profiles_path is not None
        valid_ids = {
            str(profile["profile_id"])
            for profile in self.gate_audit_payload["profiles"]
            if int(profile["counts"]["valid_remaining"]) > 0
        }
        profiles_payload = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        if "scenarios" in profiles_payload:
            profiles = [
                ObjectiveSelectionProfile(
                    profile_id=scenario.name,
                    reference_index=None,
                    request=scenario.request,
                )
                for scenario in load_verification_scenarios(self.profiles_path)
                if scenario.name in valid_ids
            ]
            if not profiles:
                raise ValueError(f"no valid profiles found in {self.gate_audit_path}")
            return profiles

        profiles = []
        for profile in profiles_payload.get("profiles", []):
            profile_id = str(profile["profile_id"])
            if profile_id not in valid_ids:
                continue
            mutable_allowed = list(profile.get("mutable_allowed", self.config.mutable_allowed))
            request = CounterfactualRequest.model_validate(
                {
                    "request_id": f"objective-sensitivity-{profile_id}",
                    "target": {
                        "target_class": self.config.target_class,
                        "min_target_probability": self.config.min_target_probability,
                    },
                    "instance": {"features": dict(profile["features"])},
                    "constraints": {"mutable_allowed": mutable_allowed},
                    "generation": {"timeout_ms": 60000},
                }
            )
            profiles.append(
                ObjectiveSelectionProfile(
                    profile_id=profile_id,
                    reference_index=(
                        int(profile["reference_index"])
                        if profile.get("reference_index") is not None
                        else None
                    ),
                    request=request,
                )
            )
        if not profiles:
            raise ValueError(f"no valid profiles found in {self.gate_audit_path}")
        return profiles

    def _input_source_payload(self) -> dict[str, Any]:
        if self.scenario_paths:
            return {
                "type": "scenario_fixtures",
                "paths": [str(path) for path in self.scenario_paths],
            }
        return {
            "type": "gate_audit",
            "gate_audit_path": str(self.gate_audit_path),
            "profiles_path": str(self.profiles_path),
        }


def _summary_for_selected(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "profile_count": len(selected),
        "mean_proximity": mean(item["proximity"] for item in selected),
        "mean_plausibility_lof": mean(item["plausibility_lof"] for item in selected),
    }


def _proximity_plausibility_overlap(profiles: list[dict[str, Any]]) -> dict[str, float | int]:
    same_count = 0
    comparable_count = 0
    for profile in profiles:
        selections = profile["selected_candidates"]
        if "proximity_only" not in selections or "plausibility_only" not in selections:
            continue
        comparable_count += 1
        if (
            selections["proximity_only"]["candidate_id"]
            == selections["plausibility_only"]["candidate_id"]
        ):
            same_count += 1
    return {
        "left_objective": "proximity_only",
        "right_objective": "plausibility_only",
        "same_candidate_count": same_count,
        "comparable_profile_count": comparable_count,
        "same_candidate_rate": same_count / comparable_count if comparable_count else 0.0,
    }
