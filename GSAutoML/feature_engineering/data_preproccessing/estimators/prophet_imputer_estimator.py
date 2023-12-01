import pandas as pd
from GSAutoML.feature_engineering.data_preproccessing.transformers.prophet_imputer_transformer import \
    ProphetImputerTransformer
from GSAutoML.feature_engineering.common.estimator_interface import IEstimator
from prophet import Prophet


class ProphetImputerEstimator(IEstimator):

    def __init__(self, timestamp_col, input_cols, seasonality_mode):
        super(ProphetImputerEstimator, self).__init__()
        self.timestamp_col = timestamp_col
        self.input_cols = input_cols
        self.seasonality_mode = seasonality_mode
        self.models = {}

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_prophet_imputer_instance()

    def fit(self, X, y=None):
        print(">>>>>>>>>> In Imputer estimator >>>>>>>>>>>>>>>>>")
        df_temp = X.copy()
        if self.timestamp_col not in df_temp.columns:
            df_temp.reset_index(inplace=True)
        df_temp.rename(columns={self.timestamp_col: 'ds'}, inplace=True)  # convert timestamp col name to ds
        for input_col in self.input_cols:
            df_impute = df_temp[['ds', input_col]]  # construct df with ds, input_col cols only
            df_impute = df_impute.rename(columns={input_col: 'y'})  # convert target col to y
            df_to_impute = df_impute.dropna()  #df without nans to fit the model
            model = Prophet(seasonality_mode=self.seasonality_mode[input_col])
            model.fit(df_to_impute)
            self.models[input_col] = model
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return ProphetImputerTransformer(
            input_cols=self.input_cols,
            timestamp_col=self.timestamp_col,
            imputer_models=self.models).transform(X)
