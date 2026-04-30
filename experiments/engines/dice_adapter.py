from __future__ import annotations

from time import perf_counter

import pandas as pd

from diabetify_cf.config import Settings
from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest, ValidationSummary
from experiments.engines.base import EngineRunResult, ExperimentEngine
from experiments.postprocessing import (
    ExperimentPostprocessor,
    ExperimentPostprocessResult,
    PreparedExperimentRequest,
)


class DiceCandidateGenerator:
    """DiCE-only candidate generator for experiment comparisons."""

    engine_version = "dice_generator_experiment_v1"

    def __init__(
        self,
        model_path: str,
        columns_path: str,
        reference_data_path: str,
        feature_registry_path: str,
    ) -> None:
        self.artifacts: ModelArtifacts | None = None
        self.initialization_error: str | None = None
        try:
            self.artifacts = load_artifacts(
                model_path=model_path,
                columns_path=columns_path,
                reference_data_path=reference_data_path,
                feature_registry_path=feature_registry_path,
            )
        except Exception as exc:
            self.initialization_error = str(exc)

    def generate_raw(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        import dice_ml

        dice_data = self._build_dice_data(postprocessor)
        dice_model = dice_ml.Model(
            model=self.artifacts.model,
            backend="sklearn",
            model_type="classifier",
        )
        dice = dice_ml.Dice(
            dice_data,
            dice_model,
            method=postprocessor.normalize_dice_method(request.generation.method),
        )
        kwargs = {
            "query_instances": prepared.query_df,
            "total_CFs": request.generation.total_cfs,
            "desired_class": postprocessor.dice_target_class(
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

        cf_examples = result.cf_examples_list[0]
        if cf_examples.final_cfs_df is None:
            return pd.DataFrame()

        raw = cf_examples.final_cfs_df.copy()
        present = [column for column in self.artifacts.feature_columns if column in raw.columns]
        if not present:
            return pd.DataFrame()
        return raw[present]

    def _build_dice_data(self, postprocessor: ExperimentPostprocessor) -> object:
        assert self.artifacts is not None
        import dice_ml

        outcome_name = "_cf_outcome"
        reference = self.artifacts.reference_data.copy()
        reference = postprocessor.as_model_input_df(reference)
        reference[outcome_name] = self.artifacts.model.predict(
            reference[self.artifacts.feature_columns]
        )
        reference = postprocessor.as_model_input_df(reference)
        continuous = list(self.artifacts.feature_columns)
        return dice_ml.Data(
            dataframe=reference,
            continuous_features=continuous,
            outcome_name=outcome_name,
        )


class DiceExperimentAdapter(ExperimentEngine):
    name = "dice"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.engine = DiceCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
        )

    def generate(self, request: CounterfactualRequest) -> EngineRunResult:
        started = perf_counter()
        if self.engine.artifacts is None:
            message = "Experiment DiCE generator is not fully configured yet."
            if self.engine.initialization_error:
                message += f" Initialization error: {self.engine.initialization_error}"
            return self._to_run_result(
                request=request,
                result=ExperimentPostprocessResult(
                    status=Status.ERROR,
                    reason_code=ReasonCode.ENGINE_NOT_READY,
                    message=message,
                    runtime_ms=self._elapsed_ms(started),
                    candidates=[],
                    input_prediction=None,
                    validation=ValidationSummary(
                        immutable_violation=False,
                        mutable_compliance=True,
                        medical_rules_passed=True,
                    ),
                ),
            )

        postprocessor = ExperimentPostprocessor(
            artifacts=self.engine.artifacts,
            max_lof_score=self.settings.max_lof_score,
        )
        try:
            prepared = postprocessor.prepare(request)
            if not prepared.mutable_allowed:
                return self._to_run_result(
                    request=request,
                    result=postprocessor.no_mutable_result(request, started),
                )
            raw_candidates = self.engine.generate_raw(
                request=request,
                prepared=prepared,
                postprocessor=postprocessor,
            )
            result = postprocessor.process(
                request=request,
                prepared=prepared,
                raw_candidates=raw_candidates,
                started=started,
            )
            return self._to_run_result(request=request, result=result)
        except ValueError as err:
            return EngineRunResult(
                engine_name=self.name,
                request_id=request.request_id,
                status=Status.ERROR.value,
                reason_code=ReasonCode.INVALID_INPUT_SCHEMA.value,
                message=str(err),
                runtime_ms=self._elapsed_ms(started),
                candidate_count=0,
                candidates=[],
                raw_error=repr(err),
            )
        except Exception as err:
            return EngineRunResult(
                engine_name=self.name,
                request_id=request.request_id,
                status=Status.ERROR.value,
                reason_code=ReasonCode.INTERNAL_ERROR.value,
                message=f"Unhandled experiment generator error: {err}",
                runtime_ms=self._elapsed_ms(started),
                candidate_count=0,
                candidates=[],
                raw_error=repr(err),
            )

    def _to_run_result(
        self,
        *,
        request: CounterfactualRequest,
        result: ExperimentPostprocessResult,
    ) -> EngineRunResult:
        return EngineRunResult(
            engine_name=self.name,
            request_id=request.request_id,
            status=result.status.value,
            reason_code=result.reason_code.value,
            message=result.message,
            runtime_ms=result.runtime_ms,
            candidate_count=len(result.candidates),
            candidates=[candidate.to_wire() for candidate in result.candidates],
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)
