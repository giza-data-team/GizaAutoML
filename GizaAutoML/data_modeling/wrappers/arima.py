import numpy as np
import pandas as pd
import time
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesARIMAModel:
    def __init__(self, value_column, p_range, d_range, q_range):
        self.value_column = value_column
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.best_model = None

    def grid_search(self, df, cv_splits=5):
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        param_grid = product(self.p_range, self.d_range, self.q_range)
        results = {
            'params': [],
            'mean_fit_time': [],
            'std_fit_time': [],
            'mean_score_time': [],
            'std_score_time': [],
            'mean_test_score': [],
            'std_test_score': [],
            'mean_train_score': [],
            'std_train_score': []
        }

        for order in param_grid:
            mae_scores = []
            fit_times = []
            score_times = []
            test_scores = []
            train_scores = []
            for train_idx, val_idx in tscv.split(df):
                train, val = df.iloc[train_idx][self.value_column], df.iloc[val_idx][self.value_column]
                try:
                    model = ARIMA(train, order=order)
                    fit_time = time.time()
                    fit_model = model.fit()
                    fit_times.append(time.time() - fit_time)

                    forecast = fit_model.forecast(steps=len(val))
                    mae = mean_absolute_error(val, forecast)
                    mae_scores.append(mae)

                    start_score_time = time.time()
                    forecast = fit_model.forecast(steps=len(val))
                    score_times.append(time.time() - start_score_time)

                    test_scores.append(mae)
                    train_scores.append(mae)

                except:
                    continue

            if len(mae_scores) > 0:
                mean_mae = np.mean(mae_scores)
                mean_fit_time = np.mean(fit_times)
                mean_score_time = np.mean(score_times)
                mean_test_score = np.mean(test_scores)
                mean_train_score = np.mean(train_scores)

                results['params'].append(order)
                results['mean_fit_time'].append(mean_fit_time)
                results['std_fit_time'].append(np.std(fit_times))
                results['mean_score_time'].append(mean_score_time)
                results['std_score_time'].append(np.std(score_times))
                results['mean_test_score'].append(round(mean_test_score, 5))
                results['std_test_score'].append(round(np.std(test_scores), 5))
                results['mean_train_score'].append(round(mean_train_score, 5))
                results['std_train_score'].append(round(np.std(train_scores), 5))

        sorted_indices = np.argsort(results['mean_test_score'])
        for key in results:
            results[key] = [results[key][i] for i in sorted_indices]

        return results

    def fit(self, df):
        grid_search_results = self.grid_search(df)
        best_order = grid_search_results['params'][0]
        self.best_model = ARIMA(df[self.value_column], order=best_order).fit()

    def predict(self, df, n_periods):
        if self.best_model is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        forecast = self.best_model.forecast(steps=n_periods)
        return forecast
