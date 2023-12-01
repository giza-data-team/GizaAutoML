import pandas as pd
from sklearn.pipeline import Pipeline

from GSAutoML.enums.stages_enum import StagesEnum
from GSAutoML.feature_engineering.feature_extraction.estimators.lagged_features_estimator import \
    LaggedFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.seasonality_features_estimator import \
    SeasonalityFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.stationary_features_estimator import \
    StationaryFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.time_features_estimator import \
    TimeFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.trend_feature_estimator import \
    TrendFeatureEstimator
from GSAutoML.feature_engineering.common.shift_target import TargetShifter
from GSAutoML.feature_engineering.feature_extraction.transformers.column_replacer import ColumnReplacer


class FeatureExtractionPipeline(Pipeline):
    def __init__(self, dataframe: pd.DataFrame, stages, series_types:dict,memory=None,
                 is_forecast=True, target_col_name="Target",timestamp_col="Timestamp"):
        self.dataframe = dataframe
        self.series_types = series_types
        self.stages_names = stages
        self.target_col_name = target_col_name
        self.timestamp_col=timestamp_col
        self.is_forecast = is_forecast
        self.stages = self._get_stages()
        self.steps = self._get_steps(self.stages)
        super().__init__(steps=self.steps, memory=memory)

        self.pipeline = Pipeline(steps=self.steps)

    def get_time_features_instance(self):
        return TimeFeaturesEstimator(timestamp_col_name=self.timestamp_col)

    def get_shifter_stage(self):
        if self.is_forecast:
            shifter = TargetShifter(self.target_col_name, -1)
        else:
            shifter = TargetShifter(self.target_col_name, 1)
        return shifter

    def get_column_replacer(self):
        return ColumnReplacer(self.target_col_name, 'Target_original')

    def get_lagged_features_instance(self):
        # apply lagged features only on target column
        return LaggedFeaturesEstimator(cols=[self.target_col_name])

    def get_seasonality_features_instance(self):
        return SeasonalityFeaturesEstimator(time_series_type=self.series_types,
                                            target_col_name=self.target_col_name,
                                            timestamp_col=self.timestamp_col)

    def get_stationary_features_instance(self):
        return StationaryFeaturesEstimator()

    def get_trend_feature_instance(self):
        # add seasonality mode of target column
        return TrendFeatureEstimator(seasonality_mode=self.series_types[self.target_col_name],
                                     target_col_name=self.target_col_name,
                                     timestamp_col=self.timestamp_col)

    def _get_stages(self) -> list:
        stages = []
        for configured_stage in self.stages_names:
            stages.append(StagesEnum[configured_stage].value.accept(self))
        print(stages)
        return stages

    def _get_steps(self, stages):
        pp_steps = [(stage.__class__.__name__ + f"_{idx}", stage) for idx, stage in enumerate(stages)]
        return pp_steps

    def fit(self, dataframe, y=None, **kwargs):
        self.pipeline = self.pipeline.fit(dataframe)
        return self

    def transform(self, dataframe):
        dataframe = self.pipeline.transform(dataframe)
        return dataframe

