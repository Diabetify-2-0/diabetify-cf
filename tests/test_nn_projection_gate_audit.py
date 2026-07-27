from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.engine.shared import ArtifactBackedCounterfactualEngine
from diabetify_cf.experiments.nn_projection_gate_audit import (
    GateAuditCounts,
    NNProjectionGateAudit,
    ProfileGateAudit,
)
from diabetify_cf.schemas import CounterfactualRequest, PredictionInfo


class _FakeModel:
    def predict_proba(self, frame: object) -> np.ndarray:
        values = frame["x"].to_numpy(dtype=float)
        low_risk = np.where(values >= 5.0, 0.8, 0.4)
        return np.column_stack([low_risk, 1.0 - low_risk])


class _TestEngine(ArtifactBackedCounterfactualEngine):
    def _generate_raw_candidates(self, **_kwargs: object) -> object:
        raise NotImplementedError


def _feature(
    name: str,
    *,
    feature_type: str = "continuous",
    global_min: float = 0.0,
    global_max: float = 10.0,
    preferred_direction: str = "any",
    allowed_transitions: dict[int, tuple[int, ...]] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type=feature_type,
        immutable=False,
        actionable=True,
        default_mutable=True,
        global_min=global_min,
        global_max=global_max,
        cost_weight=1.0,
        preferred_direction=preferred_direction,
        aliases=[],
        allowed_transitions=allowed_transitions,
    )


def test_projection_gate_audit_counts_sequential_gate_failures() -> None:
    registry = FeatureRegistry(
        version="test_v1",
        features=[
            _feature("x", preferred_direction="increase"),
            _feature(
                "category",
                feature_type="ordinal",
                global_min=0.0,
                global_max=2.0,
                allowed_transitions={0: (1,)},
            ),
            _feature("binary", feature_type="binary", global_min=0.0, global_max=1.0),
        ],
    )
    engine = _TestEngine.__new__(_TestEngine)
    engine.artifacts = SimpleNamespace(model=_FakeModel())
    engine._as_model_input_df = lambda frame: frame
    audit = NNProjectionGateAudit.__new__(NNProjectionGateAudit)
    audit.engine = engine
    audit.artifacts = engine.artifacts
    request = CounterfactualRequest.model_validate(
        {
            "request_id": "gate-audit-test",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {"features": {"x": 0.0, "category": 0, "binary": 0}},
            "constraints": {"mutable_allowed": ["x", "category", "binary"]},
        }
    )
    prepared = SimpleNamespace(
        instance_features={"x": 0.0, "category": 0, "binary": 0},
        mutable_allowed=["x", "category", "binary"],
        registry=registry,
        model_columns=["x", "category", "binary"],
        base_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.2),
    )
    candidates = [
        {"x": -1.0, "category": 1, "binary": 0},
        {"x": 1.0, "category": 2, "binary": 0},
        {"x": 1.0, "category": 1, "binary": 2},
        {"x": 1.0, "category": 1, "binary": 1},
        {"x": 5.0, "category": 1, "binary": 1},
    ]

    counts = audit._audit_candidates(
        raw_generated_candidates=6,
        candidates=candidates,
        request=request,
        prepared=prepared,
    )

    assert counts.raw_generated_candidates == 6
    assert counts.duplicate_candidates_removed == 1
    assert counts.total_candidates == 5
    assert counts.failed_directional == 1
    assert counts.failed_transition == 1
    assert counts.failed_medical == 1
    assert counts.failed_target == 1
    assert counts.valid_remaining == 1


def test_profile_gate_audit_payload_omits_null_reference_index() -> None:
    audit = ProfileGateAudit(
        profile_id="profile-1",
        reference_index=None,
        baseline_probability_low_risk=0.2,
        counts=GateAuditCounts(
            raw_generated_candidates=1,
            duplicate_candidates_removed=0,
            total_candidates=1,
            failed_directional=0,
            failed_transition=0,
            failed_medical=0,
            failed_target=0,
            valid_remaining=1,
        ),
    )

    payload = audit.to_payload()

    assert "reference_index" not in payload
