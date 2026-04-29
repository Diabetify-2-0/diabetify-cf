from __future__ import annotations

from typing import Any

from diabetify_cf.engine.feature_registry import FeatureRegistry
from diabetify_cf.schemas import CounterfactualRequest


def _as_float(value: Any) -> float:
    return float(value)


def _equal(a: Any, b: Any) -> bool:
    try:
        return abs(_as_float(a) - _as_float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def _changed_features(candidate: dict[str, Any], baseline: dict[str, Any]) -> set[str]:
    changed: set[str] = set()
    for feature_name, value in candidate.items():
        if feature_name not in baseline:
            continue
        if not _equal(value, baseline[feature_name]):
            changed.add(feature_name)
    return changed


def _target_success(candidate: dict[str, Any], request: CounterfactualRequest) -> bool:
    prediction = candidate.get("prediction", {})
    if not isinstance(prediction, dict):
        return False

    target_class = request.target.target_class.strip().lower()
    predicted_class = str(prediction.get("class", "")).strip().lower()
    probability_low_risk = float(prediction.get("probability_low_risk", 0.0))

    if target_class in {"low", "low_risk", "non_diabetes", "0"}:
        return (
            predicted_class == "low_risk"
            and probability_low_risk >= request.target.min_target_probability
        )
    if target_class in {"high", "high_risk", "diabetes", "1"}:
        return (
            predicted_class == "high_risk"
            and (1.0 - probability_low_risk) >= request.target.min_target_probability
        )
    return probability_low_risk >= request.target.min_target_probability


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    request: CounterfactualRequest,
    registry: FeatureRegistry,
    max_lof_score: float,
) -> dict[str, Any]:
    features = candidate.get("features", {})
    if not isinstance(features, dict):
        features = {}

    canonical_baseline = registry.canonicalize_feature_map(baseline)
    immutable = set(registry.immutable_defaults())
    immutable.update(registry.canonicalize_feature_names(request.constraints.immutable_features))
    immutable.update(registry.canonicalize_feature_names(request.constraints.must_not_change))
    mutable = set(registry.canonicalize_feature_names(request.constraints.mutable_allowed))
    changed = _changed_features(features, canonical_baseline)

    immutable_violations = changed.intersection(immutable)
    mutable_violations = {feature for feature in changed if feature not in mutable}

    bounds_violations: set[str] = set()
    for raw_name, bound in request.constraints.feature_bounds.items():
        feature_name = registry.resolve_name(raw_name)
        if feature_name not in features:
            continue
        value = _as_float(features[feature_name])
        if bound.min is not None and value < bound.min:
            bounds_violations.add(feature_name)
        if bound.max is not None and value > bound.max:
            bounds_violations.add(feature_name)

    directional_violations: set[str] = set()
    for feature_name in changed.intersection(mutable):
        feature = registry.get(feature_name)
        if feature is None or feature.preferred_direction == "any":
            continue
        if feature_name not in canonical_baseline:
            continue
        delta = _as_float(features[feature_name]) - _as_float(canonical_baseline[feature_name])
        if feature.preferred_direction == "increase" and delta < 0:
            directional_violations.add(feature_name)
        if feature.preferred_direction == "decrease" and delta > 0:
            directional_violations.add(feature_name)

    metrics = candidate.get("metrics", {})
    lof_score = float(metrics.get("lof_score", 1.0)) if isinstance(metrics, dict) else 1.0

    return {
        "target_success": _target_success(candidate, request),
        "immutable_violation_count": len(immutable_violations),
        "mutable_violation_count": len(mutable_violations),
        "bounds_violation_count": len(bounds_violations),
        "directional_violation_count": len(directional_violations),
        "plausibility_pass": lof_score <= max_lof_score,
        "changed_feature_count_eval": len(changed),
    }
