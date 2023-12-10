# Define a wrapper class for Prophet
import time

from prophet import Prophet
from sklearn.base import BaseEstimator
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import numpy as np
import pandas as pd

class SkProphet(BaseEstimator):
    def __init__(self, label_col, time_stamp_col_name, seasonality_mode='multiplicative'):
        self.seasonality_mode = seasonality_mode
        self.time_stamp_col_name = time_stamp_col_name
        self.label_col = label_col
        self.params = None

    def fit(self, X, y=None):
        # X should be a DataFrame with 'ds' and 'y' columns
        pd = X.copy()
        pd.rename(columns={self.time_stamp_col_name: 'ds'}, inplace=True)  # convert timestamp col name to ds
        pd.rename(columns={self.label_col: 'y'}, inplace=True)
        # self.model = Prophet(seasonality_mode=self.seasonality_mode)
        # print(self.params)
        #def _get_best_model() #TODO
        # self.model.fit(pd)
        self.model = self._get_best_model(pd)
        return self.model, self.grid_search_results, self.best_params

    def transform(self, X):
        X.rename(columns={self.time_stamp_col_name: 'ds'}, inplace=True)
        X['yhat'] = self.predict(x=X.drop(columns=[self.label_col], axis=1))\
            if self.label_col in list(X.columns) else self.predict(x=X)
        return X

    def predict(self, x):
        # X should be a DataFrame with 'ds' column
        x.rename(columns={self.time_stamp_col_name: 'ds'}, inplace=True)
        forecast = self.model.predict(x)
        return forecast['yhat']

    def _get_best_model(self, train_data):
        """
        apply grid search on estimator to get the best hyperparameters
        :param x: dataframe containing the features to make predictions based on
        :param y: the target col
        :return: best estimator
        """
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search_results = {'params': [],
                               'mean_fit_time': [],
                               'std_fit_time': [],
                               'mean_score_time': [],
                               'std_score_time': [],
                               'mean_test_score': [],
                               'std_test_score': [],
                               'mean_train_score': [],
                               'std_train_score': []
                               }
        results = []
        # Loop through all combinations of hyperparameters
        for params in product(*self.params.values()):
            hyperparameters = dict(zip(self.params.keys(), params))
            test_scores = []
            fit_times = []
            score_times = []
            test_scores = []
            train_scores = []
            for train_idx, test_idx in tscv.split(train_data):
                train_fold = train_data.iloc[train_idx]
                test_fold = train_data.iloc[test_idx]
                start_time = time.time()
                model = Prophet(**hyperparameters)
                model.fit(train_fold)
                fit_times.append(time.time() - start_time)
                start_time = time.time()
                predictions = model.predict(test_fold)
                score_times.append(time.time() - start_time)
                mean_test_score = mean_absolute_error(test_fold['y'], predictions['yhat'])
                test_scores.append(mean_test_score)
                results.append((hyperparameters, mean_test_score))
                train_scores.append(mean_absolute_error(train_data['y'], model.predict(train_data)['yhat']))
            grid_search_results['params'].append(hyperparameters)
            grid_search_results['mean_fit_time'].append(np.mean(fit_times))
            grid_search_results['std_fit_time'].append(np.std(fit_times))
            grid_search_results['mean_score_time'].append(np.mean(score_times))
            grid_search_results['std_score_time'].append(np.std(score_times))
            grid_search_results['mean_test_score'].append(np.mean(test_scores))
            grid_search_results['std_test_score'].append(np.std(test_scores))
            grid_search_results['mean_train_score'].append(np.mean(train_scores))
            grid_search_results['std_train_score'].append(np.std(train_scores))
        self.best_params, best_score = min(results, key=lambda x: x[1])
        best_model = Prophet(**self.best_params)
        best_model.fit(train_data)
        self.grid_search_results = pd.DataFrame.from_dict(grid_search_results)
        return best_model

