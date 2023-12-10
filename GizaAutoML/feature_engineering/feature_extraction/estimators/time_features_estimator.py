import pandas as pd

from GizaAutoML.feature_engineering.common.estimator_interface import IEstimator
from GizaAutoML.feature_engineering.feature_extraction.transformers.time_features_transformer import \
    TimeFeaturesTransformer


class TimeFeaturesEstimator(IEstimator):

    def __init__(self, timestamp_col_name="Timestamp"):
        super().__init__()
        self.timestamp_col_name = timestamp_col_name

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_time_features_instance()

    def fit(self, dataframe: pd.DataFrame, y=None):
        print(">>>>>>>>>> In Time Features estimator >>>>>>>>>>>>>>>>>")

        self._schema_extracted_features = {'DayOfWeek': 'DayOfWeek',
                                           'MonthOfYear': 'MonthOfYear',
                                           'HourOfDay': 'HourOfDay'}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return TimeFeaturesTransformer(self.timestamp_col_name).transform(X)
