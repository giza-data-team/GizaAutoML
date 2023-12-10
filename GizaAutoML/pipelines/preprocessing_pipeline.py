from sklearn.pipeline import Pipeline
from GizaAutoML.feature_engineering.data_preproccessing.estimators.prophet_imputer_estimator import \
    ProphetImputerEstimator
from GizaAutoML.feature_engineering.data_preproccessing.estimators.constant_columns_removal_estimator import \
    ConstantColumnsRemovalEstimator
from GizaAutoML.feature_engineering.data_preproccessing.estimators.encoder_estimator import \
    EncoderEstimator
from GizaAutoML.enums.stages_enum import StagesEnum
from GizaAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer import \
    MinMaxScalerTransformer
from GizaAutoML.feature_engineering.data_preproccessing.transformers.Log_transformer import LogTransformer


class PreprocessingPipeline(Pipeline):
    def __init__(self, stages, excluded_columns: list = None,
                 input_columns:list=[], series_types: dict = None, memory=None):

        self.series_types = series_types
        self.target_col_name = "Target"
        self.timestamp_col_name = "Timestamp"
        self.input_columns = input_columns
        self.excluded_columns = excluded_columns if excluded_columns else []
        self.stages_names = stages
        self.stages = self._get_stages()
        super().__init__(steps=self._get_steps(self.stages), memory=memory)
        # self.steps = self._get_steps(self.stages)
        self.pipeline = Pipeline(steps=self.steps)

    def get_prophet_imputer_instance(self):
        return ProphetImputerEstimator(timestamp_col=self.timestamp_col_name,
                                       input_cols=self.input_columns,
                                       seasonality_mode=self.series_types)

    def get_consant_columns_removal(self):
        return ConstantColumnsRemovalEstimator()

    def get_encoder_instance(self):
        return EncoderEstimator()

    def get_normalizer_instance(self):
        return MinMaxScalerTransformer(excluded_columns=self.excluded_columns)

    def get_log_transformer_instance(self):
        return LogTransformer(column_name = self.target_col_name)

    def _get_stages(self):
        """ get pipeline stages """
        stages = []
        for configured_stage in self.stages_names:
            stages.append(StagesEnum[configured_stage].value.accept(self))
        return stages

    def _get_steps(self, stages):
        pp_steps = [(stage.__class__.__name__ + f"_{idx}", stage) for idx, stage in enumerate(stages)]
        return pp_steps

    def fit(self, X, y=None, **kwargs):
        pipeline = Pipeline(steps=self.steps)
        self.pipeline = pipeline.fit(X=X, y=y)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return X
