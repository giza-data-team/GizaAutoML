from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer
from prophet import Prophet
import pandas as pd


class ProphetImputerTransformer(ITransformer):
    def __init__(self, input_cols, timestamp_col, imputer_models):
        super(ProphetImputerTransformer, self).__init__()
        self.input_cols = input_cols
        self.timestamp_col = timestamp_col
        self.set_imputer_model(model=imputer_models)

    def set_imputer_model(self, model):
        self.imputer_models = model

    def get_imputer_model(self):
        return self.imputer_models

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_temp = X.copy()
        if self.timestamp_col not in list(df_temp.columns):
            df_temp.reset_index(inplace=True)

        df_temp.rename(columns={self.timestamp_col: 'ds'}, inplace=True)  # convert timestamp col name to ds
        for input_col in self.input_cols:
            df_impute = df_temp[['ds', input_col]]  # construct df with ds, input_col cols only
            print("no of missing values: ", df_impute[input_col].isnull().sum())
            if df_impute[input_col].isnull().sum() > 0:
                nan_values = df_impute[df_impute[input_col].isnull()]  # save rows with nan values
                forecast = self.imputer_models[input_col].predict(nan_values['ds'].to_frame())
                df_impute.iloc[list(nan_values.index), [1]] = forecast['yhat'] # Merge imputed values back into the original DataFrame
                df_temp[input_col] = df_impute[input_col]
        df_temp.rename(columns={'ds': self.timestamp_col}, inplace=True)
        return df_temp
