from __future__ import annotations

import pandas as pd

from experiments.engines.dice_adapter import DiceCandidateGenerator


def test_dice_no_counterfactual_error_returns_empty_candidates() -> None:
    generator = DiceCandidateGenerator(
        model_path="missing-model.pkl",
        columns_path="missing-columns.pkl",
        reference_data_path="missing-reference.parquet",
        feature_registry_path="missing-registry.json",
    )

    assert generator._is_no_counterfactual_error(
        Exception(
            "No counterfactuals found for any of the query points! Kindly check your configuration."
        )
    )


def test_dice_non_matching_error_is_not_treated_as_empty_candidates() -> None:
    generator = DiceCandidateGenerator(
        model_path="missing-model.pkl",
        columns_path="missing-columns.pkl",
        reference_data_path="missing-reference.parquet",
        feature_registry_path="missing-registry.json",
    )

    assert not generator._is_no_counterfactual_error(Exception("Some other engine failure"))
