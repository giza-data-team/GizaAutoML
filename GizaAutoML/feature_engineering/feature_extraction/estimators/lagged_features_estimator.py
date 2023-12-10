import pandas as pd
from statsmodels.tsa.stattools import pacf

from GizaAutoML.feature_engineering.common.estimator_interface import IEstimator
from GizaAutoML.feature_engineering.feature_extraction.transformers.lagged_features_transformer import \
    LaggedFeaturesTransformer
import numpy as np
import math

class LaggedFeaturesEstimator(IEstimator):

    def __init__(self, n_lags=None, cols: list = None):
        super().__init__()
        self.cols = cols
        self.transformer = None
        self.consecutive_non_significant_lags_no = 3
        self.lags_threshold = 30
        self.target_col_name = "Target"
        self.n_lags = self.__n_lags(n_lags)

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_lagged_features_instance()

    def set_num_lagged_features(self, lagged_features_dict):
        self.num_lagged_features = len(lagged_features_dict)

    def get_num_lagged_features(self):
        return self.num_lagged_features

    def set_col_with_lags(self,col_with_lags):
        self.col_with_lags = col_with_lags

    def get_col_with_lags(self):
        return self.col_with_lags
    def set_pacf_result(self, pacf_result):
        self.pacf_result = pacf_result

    def get_pacf_result(self):
        return self.pacf_result

    def fit(self, df: pd.DataFrame, y=None):
        print(">>>>>>>>>> In lagger estimator >>>>>>>>>>>>>>>>>")
        # shift target col
        dataframe = df.copy()

        if math.floor(dataframe.shape[0]/2) <= self.lags_threshold:
            self.lags_threshold = int(dataframe.shape[0]/2 - 1)
        # apply in stationary version of Target column if exist
        cols_names = self.get_input_col_names(list(dataframe.columns)) if self.cols is None else self.cols
        cols_names = [col for col in cols_names if dataframe.dtypes[col] not in ["object", "datetime64[ns]"]]

        pacf_result = {}
        lags_dict = {'last_significant_lag_index': -1,
                     'non_significant_lags_indices': []}
        col_lags_dic = {}
        for col in cols_names:
            pacf_magnitude, conf_int = self.__calc_pacf(dataframe, col, self.lags_threshold)
            consecutive_non_significant_lags_count = 0
            for lag in range(1, self.lags_threshold + 1):
                if (np.min(conf_int[lag]) < 0) and (np.max(conf_int[lag]) > 0):
                    consecutive_non_significant_lags_count += 1
                    if consecutive_non_significant_lags_count == \
                            self.consecutive_non_significant_lags_no:
                        # Stop when n consecutive non-significant lags found
                        last_sig_lag_index = lag - consecutive_non_significant_lags_count
                        break

                    lags_dict['non_significant_lags_indices'].append(lag)
                else:
                    consecutive_non_significant_lags_count = 0

                if lag == self.lags_threshold:
                    last_sig_lag_index = lag - consecutive_non_significant_lags_count

            lags_dict['last_significant_lag_index'] = last_sig_lag_index
            pacf_result[col] = pacf_magnitude.tolist()
            col_lags_dic[col] = lags_dict
        self.col_lags_dic = col_lags_dic
        self.set_pacf_result(pacf_result)
        col_with_lags = []
        for col in col_lags_dic.keys():
            col_lags = col_lags_dic[col]

            # get number of significant lags not the index
            if len(col_lags['non_significant_lags_indices']):
                if col_lags['last_significant_lag_index'] < col_lags['non_significant_lags_indices'][-1]:
                    significant_lags_no = (col_lags['last_significant_lag_index'] + 1) - \
                                          (len(col_lags['non_significant_lags_indices']) - (
                                                      self.consecutive_non_significant_lags_no - 1))
                else:
                    significant_lags_no = (col_lags['last_significant_lag_index'] + 1) - \
                                          len(col_lags['non_significant_lags_indices'])

            else:
                significant_lags_no = col_lags['last_significant_lag_index'] + 1
            col_lags['significant_lags'] = significant_lags_no
            col_lags['non_significant_lags_count'] = sum([1 for x in col_lags['non_significant_lags_indices'] if x < col_lags['last_significant_lag_index']])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transformer = LaggedFeaturesTransformer(lagged_features=self.col_lags_dic,
                                                     pacf_result=self.pacf_result)
        return self.transformer.transform(X)

    def __n_lags(self, n_lags):
        if n_lags is None:
            return self.lags_threshold
        else:
            return n_lags

    @staticmethod
    def __calc_pacf(dataframe, col_name, n_lags):
        """
        Compute pACF for a specific column with number of lags.

        :param col_name: str - The name of the column.
        :param n_lags : int - Number of Lags.
        :returns Tuple(ndarray: The partial auto correlations for lags 0, 1, …, n_lags. Shape (n_lags+1,),
        ndarray: Confidence intervals for the pACF at lags 0, 1, …, nlags. Shape (nlags + 1, 2)).
        """
        values = pacf(dataframe[col_name].to_numpy(), nlags=n_lags, method='ols', alpha=.05)
        pacf_lags = values[0]
        conf_int = values[1]
        return pacf_lags, conf_int

    def get_input_col_names(self, cols):
        cols_names = [col for col in cols if col.startswith(self.target_col_name + '_diff_')]
        if cols_names:
            return cols_names
        else:
            return [self.target_col_name]
