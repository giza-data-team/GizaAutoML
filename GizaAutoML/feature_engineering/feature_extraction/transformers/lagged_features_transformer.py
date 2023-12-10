import pandas as pd
from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer


class LaggedFeaturesTransformer(ITransformer):
    def __init__(self, lagged_features: dict = None, pacf_result: dict = None):
        super().__init__()
        self.set_lagged_features(lagged_features)
        self.set_pacf_result(pacf_result)
        self.target_col_name = "Target"

    def set_lagged_features(self, lagged_features):
        self.lagged_features = lagged_features

    def get_lagged_features(self):
        return self.lagged_features

    def set_pacf_result(self, pacf_result):
        self.pacf_result = pacf_result

    def get_pacf_result(self):
        return self.pacf_result

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(">>>>>>>>>> In lagger transformer >>>>>>>>>>>>>>>>>")
        # shift target col
        dataframe = df.copy()

        features_dict = dict()
        for col in self.get_lagged_features().keys():
            col_lags = self.get_lagged_features()[col]
            for lag in range(1, col_lags['last_significant_lag_index']+1):
                if lag not in col_lags['non_significant_lags_indices']:
                    features_dict[f"{col}_lag_{lag}"] = dataframe[col].shift(lag).to_list()
        df_added = pd.DataFrame(features_dict, index=dataframe.index.values)
        all_df = pd.concat([dataframe, df_added], axis=1)
        all_df = all_df.dropna()
        all_df.reset_index(inplace=True, drop=True)
        # remove stationary version of Target column if exist
        cols_names = [col for col in list(all_df.columns) if
                      col.startswith(self.target_col_name + '_diff_') and "lag" not in col]
        if cols_names:
            all_df = all_df.drop(columns=cols_names, axis=1)
        return all_df
