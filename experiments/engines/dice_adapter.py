from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from diabetify_cf.config import Settings
from diabetify_cf.engine.artifacts import ModelArtifacts, load_artifacts
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    ValidationSummary,
)
from experiments.engines.base import EngineRunResult, ExperimentEngine
from experiments.postprocessing import (
    ExperimentPostprocessor,
    ExperimentPostprocessResult,
    PreparedExperimentRequest,
)


class DiceCandidateGenerator:
    """DiCE-backed candidate generator used by research experiment adapters."""

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
        self._dice_data_cache: Any | None = None
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
        use_native_constraints: bool,
    ) -> pd.DataFrame:
        assert self.artifacts is not None
        import dice_ml

        dice_data = self._get_dice_data(postprocessor)
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
            "verbose": False,
            "random_seed": request.generation.random_seed,
        }
        if use_native_constraints:
            kwargs["features_to_vary"] = prepared.mutable_allowed
        if use_native_constraints and prepared.permitted_range:
            kwargs["permitted_range"] = prepared.permitted_range

        try:
            result = self._invoke_dice(dice, kwargs)
        except Exception as exc:
            if self._is_no_counterfactual_error(exc):
                return pd.DataFrame()
            raise

        cf_examples = result.cf_examples_list[0]
        if cf_examples.final_cfs_df is None:
            return pd.DataFrame()

        raw = cf_examples.final_cfs_df.copy()
        present = [column for column in self.artifacts.feature_columns if column in raw.columns]
        if not present:
            return pd.DataFrame()
        return raw[present]

    @staticmethod
    def _invoke_dice(dice: Any, kwargs: dict[str, Any]) -> Any:
        try:
            return dice.generate_counterfactuals(**kwargs)
        except TypeError:
            retry_kwargs = {key: value for key, value in kwargs.items() if key != "random_seed"}
            return dice.generate_counterfactuals(**retry_kwargs)

    @staticmethod
    def _is_no_counterfactual_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "no counterfactuals found" in message

    def _get_dice_data(self, postprocessor: ExperimentPostprocessor) -> object:
        if self._dice_data_cache is None:
            self._dice_data_cache = self._build_dice_data(postprocessor)
        return self._dice_data_cache

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
    name = "dice_constrained_native"
    apply_production_gates = False

    def __init__(
        self,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        _ = config
        self.settings = settings or Settings()
        self.engine = DiceCandidateGenerator(
            model_path=self.settings.model_path,
            columns_path=self.settings.columns_path,
            reference_data_path=self.settings.reference_data_path,
            feature_registry_path=self.settings.feature_registry_path,
        )

    @property
    def artifacts(self) -> ModelArtifacts | None:
        return self.engine.artifacts

    @property
    def initialization_error(self) -> str | None:
        return self.engine.initialization_error

    def generate(self, request: CounterfactualRequest) -> EngineRunResult:
        started = perf_counter()
        if self.artifacts is None:
            message = "Experiment DiCE generator is not fully configured yet."
            if self.initialization_error:
                message += f" Initialization error: {self.initialization_error}"
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
            artifacts=self.artifacts,
            max_lof_score=self.settings.max_lof_score,
        )
        try:
            prepared = postprocessor.prepare(request)
            if not prepared.mutable_allowed:
                return self._to_processed_run_result(
                    request=request,
                    result=postprocessor.no_mutable_result(request, started),
                )
            raw_candidates = self.engine.generate_raw(
                request=request,
                prepared=prepared,
                postprocessor=postprocessor,
                use_native_constraints=True,
            )
            if self.apply_production_gates:
                result = postprocessor.process(
                    request=request,
                    prepared=prepared,
                    raw_candidates=raw_candidates,
                    started=started,
                )
                return self._to_processed_run_result(request=request, result=result)
            return self._to_raw_run_result(
                request=request,
                prepared=prepared,
                postprocessor=postprocessor,
                raw_candidates=raw_candidates,
                started=started,
            )
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

    def _to_raw_run_result(
        self,
        *,
        request: CounterfactualRequest,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
        raw_candidates: pd.DataFrame,
        started: float,
    ) -> EngineRunResult:
        if raw_candidates.empty:
            result = ExperimentPostprocessResult(
                status=Status.INFEASIBLE,
                reason_code=ReasonCode.TARGET_UNREACHABLE_UNDER_CONSTRAINTS,
                message="DiCE did not return any candidate under the current native setup.",
                runtime_ms=self._elapsed_ms(started),
                candidates=[],
                input_prediction=prepared.base_prediction,
                validation=ValidationSummary(
                    immutable_violation=False,
                    mutable_compliance=True,
                    medical_rules_passed=True,
                ),
            )
            return self._to_processed_run_result(request=request, result=result)

        candidates = [
            self._candidate_from_raw_row(
                candidate_id=f"cf_{index + 1}",
                row=row,
                prepared=prepared,
                postprocessor=postprocessor,
            )
            for index, (_, row) in enumerate(raw_candidates.iterrows())
        ]
        return EngineRunResult(
            engine_name=self.name,
            request_id=request.request_id,
            status=Status.FEASIBLE.value,
            reason_code=ReasonCode.OK.value,
            message=f"Generated {len(candidates)} raw DiCE candidate(s) without pipeline filtering.",
            runtime_ms=self._elapsed_ms(started),
            candidate_count=len(candidates),
            candidates=[candidate.to_wire() for candidate in candidates],
        )

    def _to_processed_run_result(
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

    def _candidate_from_raw_row(
        self,
        *,
        candidate_id: str,
        row: pd.Series,
        prepared: PreparedExperimentRequest,
        postprocessor: ExperimentPostprocessor,
    ) -> CounterfactualCandidate:
        candidate_features: dict[str, int | float | bool | str] = {}
        for column in prepared.model_columns:
            raw_value = row[column] if column in row.index else prepared.instance_features[column]
            if pd.isna(raw_value):
                raw_value = prepared.instance_features[column]
            candidate_features[column] = prepared.registry.coerce_value(column, raw_value)

        candidate_df = pd.DataFrame([candidate_features], columns=prepared.model_columns)
        candidate_df = postprocessor.as_model_input_df(candidate_df)
        prediction = postprocessor.predict_info(candidate_df)
        delta = self._build_full_delta(
            candidate_features=candidate_features,
            baseline_features=prepared.instance_features,
        )
        distance_l1 = postprocessor._normalized_l1(  # noqa: SLF001
            candidate_features,
            prepared.instance_features,
            prepared.registry,
        )
        lof_score = postprocessor._lof_score(candidate_df)  # noqa: SLF001
        return CounterfactualCandidate(
            candidate_id=candidate_id,
            features=candidate_features,
            delta=delta,
            prediction=prediction,
            metrics=CandidateMetrics(
                distance_l1=distance_l1,
                changed_feature_count=len(delta),
                lof_score=lof_score,
                constraint_violations=0,
            ),
        )

    @staticmethod
    def _build_full_delta(
        *,
        candidate_features: dict[str, int | float | bool | str],
        baseline_features: dict[str, int | float | bool | str],
    ) -> dict[str, float]:
        delta: dict[str, float] = {}
        for feature_name, candidate_value in candidate_features.items():
            if feature_name not in baseline_features:
                continue
            diff = float(candidate_value) - float(baseline_features[feature_name])
            if abs(diff) >= 1e-9:
                delta[feature_name] = round(diff, 6)
        return delta

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)


class DiceConstrainedGatedExperimentAdapter(DiceExperimentAdapter):
    name = "dice_constrained_gated"
    apply_production_gates = True
