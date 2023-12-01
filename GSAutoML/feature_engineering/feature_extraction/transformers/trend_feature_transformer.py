import pandas as pd
from GSAutoML.feature_engineering.common.transformer_interface import ITransformer


class TrendFeatureTransformer(ITransformer):

    def __init__(self, target_col_name,timestamp_col, trend_model, trend_type, cap):
        super().__init__()
        self.target_col_name = target_col_name
        self.trend_type = trend_type
        self.timestamp_col_name = timestamp_col
        self.cap = cap
        self.set_trend_model(model=trend_model)

    def set_trend_model(self, model):
        self.trend_model = model

    def get_trend_model(self):
        return self.trend_model

    def fit(self, X, y=None):
        return self

    def transform(self, dataframe: pd.DataFrame):
        print(">>>>>>>>>> In trend transformer >>>>>>>>>>>>>>>>>")

        df = dataframe.copy()
        trend_col_name = "Trend"
        X = df[[self.timestamp_col_name, self.target_col_name]]

        X.rename(columns={self.timestamp_col_name: 'ds',
                          self.target_col_name: 'y'}, inplace=True)
        # add cap column in case of logistic trend
        if self.trend_type == "logistic":
            X['cap'] = self.cap

        df[trend_col_name] = self.get_trend_model().predict(X)['trend']

        return df

