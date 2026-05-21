from __future__ import annotations

import pandas as pd

from diabetify_cf.engine.shared import ArtifactBackedCounterfactualEngine, PreparedRequest
from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.schemas import CounterfactualRequest, PredictionInfo


class DiceCounterfactualEngine(ArtifactBackedCounterfactualEngine):
    engine_version = "dice_engine_v4"

    def __init__(
        self,
        *,
        model_path: str = "",
        columns_path: str = "",
        reference_data_path: str = "",
        feature_registry_path: str = "",
        max_lof_score: float = 2.5,
        planner: PrescriptivePlanner | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            columns_path=columns_path,
            reference_data_path=reference_data_path,
            feature_registry_path=feature_registry_path,
            max_lof_score=max_lof_score,
            planner=planner,
        )
        self._dice_data_cache: object | None = None

    def _generate_raw_candidates(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedRequest,
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        import dice_ml

        dice_data = self._get_dice_data()
        dice_model = dice_ml.Model(
            model=self.artifacts.model, backend="sklearn", model_type="classifier"
        )
        dice = dice_ml.Dice(
            dice_data,
            dice_model,
            method=self._normalize_dice_method(request.generation.method),
        )

        kwargs = {
            "query_instances": prepared.query_df,
            "total_CFs": request.generation.total_cfs,
            "desired_class": self._target_to_dice_class(
                request.target.target_class,
                prepared.base_prediction,
            ),
            "features_to_vary": prepared.mutable_allowed,
            "verbose": False,
            "random_seed": request.generation.random_seed,
        }
        if prepared.permitted_range:
            kwargs["permitted_range"] = prepared.permitted_range

        try:
            result = dice.generate_counterfactuals(**kwargs)
        except TypeError:
            kwargs.pop("random_seed", None)
            result = dice.generate_counterfactuals(**kwargs)
        except Exception as exc:
            if "no counterfactuals found" in str(exc).lower():
                return pd.DataFrame()
            raise

        cf_examples = result.cf_examples_list[0]
        if cf_examples.final_cfs_df is None:
            return pd.DataFrame()

        raw = cf_examples.final_cfs_df.copy()
        model_columns = self.artifacts.feature_columns
        present = [column for column in model_columns if column in raw.columns]
        if not present:
            return pd.DataFrame()
        return raw[present]

    def _get_dice_data(self) -> object:
        if self._dice_data_cache is None:
            self._dice_data_cache = self._build_dice_data()
        return self._dice_data_cache

    def _build_dice_data(self) -> object:
        assert self.artifacts is not None
        import dice_ml

        outcome_name = "_cf_outcome"
        reference = self.artifacts.reference_data.copy()
        reference = self._as_model_input_df(reference)
        reference[outcome_name] = self.artifacts.model.predict(
            reference[self.artifacts.feature_columns]
        )
        reference = self._as_model_input_df(reference)
        continuous = list(self.artifacts.feature_columns)

        return dice_ml.Data(
            dataframe=reference,
            continuous_features=continuous,
            outcome_name=outcome_name,
        )

    def _as_dice_input_df(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._as_model_input_df(frame)

    def _target_to_dice_class(self, target: str, base_prediction: PredictionInfo) -> int | str:
        normalized = target.strip().lower()
        if normalized in {"low", "low_risk", "non_diabetes", "0"}:
            return 0
        if normalized in {"high", "high_risk", "diabetes", "1"}:
            return 1
        if base_prediction.class_name == "low_risk":
            return 1
        return 0

    def _normalize_dice_method(self, method: str) -> str:
        mapping = {
            "dice_genetic": "genetic",
            "genetic": "genetic",
            "dice_random": "random",
            "random": "random",
            "dice_kdtree": "kdtree",
            "kdtree": "kdtree",
        }
        return mapping.get(method.lower(), "genetic")
