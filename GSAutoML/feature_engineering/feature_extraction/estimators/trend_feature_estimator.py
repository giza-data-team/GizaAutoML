import pandas as pd
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from GSAutoML.feature_engineering.common.estimator_interface import IEstimator
from GSAutoML.feature_engineering.feature_extraction.transformers.trend_feature_transformer import \
    TrendFeatureTransformer
from GSAutoML.enums.time_series_types_enum import TimeSeriesTypesEnum


class TrendFeatureEstimator(IEstimator):

    def __init__(self, seasonality_mode=None, target_col_name=None, timestamp_col=None): #"Timestamp"
        super().__init__()
        self.seasonality_mode = seasonality_mode if seasonality_mode else TimeSeriesTypesEnum.ADDITIVE.value # seasonality mode for target attribute
        self.target_col_name = target_col_name if target_col_name else "Target"
        self.trend_type = None
        self.timestamp_col = timestamp_col
        self.train_cap=None

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_trend_feature_instance()

    def get_trend_type(self,series_df)-> str:
        """
         check trend type (Linear or logistic) using adfuller test
        :param series_df: series to check the type of its trend
        :returns: str defines series type (linear or logistic)
        """
        adf_result = adfuller(series_df)
        p_value= adf_result[1]
        if p_value < 0.05:
            return "linear"
        else:
             return "logistic"

    def fit(self, dataframe: pd.DataFrame, y=None):
        """
        Apply prophet on series to get the trend of the series,
        apply only on the target column
        """
        print(">>>>>>>>>> In Trend estimator >>>>>>>>>>>>>>>>>")

        try:
            X = dataframe[[self.timestamp_col, self.target_col_name]]
        except KeyError:
            X = dataframe.reset_index()[[self.timestamp_col, self.target_col_name]]

        # Prophet requires A DataFrame containing 'ds' column for timestamps and 'y' column for values
        X.rename(columns={self.timestamp_col:'ds',
                          self.target_col_name:'y'}, inplace=True)
        self.trend_type = self.get_trend_type(X.set_index('ds'))
        if self.trend_type == "logistic":
            # with a logistic growth trend, we need to provide information about the capacity (maximum limit) that the time series is expected to approach as it grows
            self.train_cap = max(X.y.values)
            X['cap'] = self.train_cap

        print(f"Trend Type:{self.trend_type} ")
        self.trend_model = Prophet(seasonality_mode=self.seasonality_mode,
                                   growth=self.trend_type)
        self.trend_model.fit(X)
        return self

    def transform(self, dataframe: pd.DataFrame)-> pd.DataFrame:
       return TrendFeatureTransformer(target_col_name=self.target_col_name,
                                      trend_model=self.trend_model,
                                      trend_type=self.trend_type,
                                      cap=self.train_cap,
                                      timestamp_col=self.timestamp_col).transform(dataframe)

