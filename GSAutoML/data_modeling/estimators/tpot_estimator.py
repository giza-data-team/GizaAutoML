# from tpot.tpot import TPOTRegressor
from tpot import TPOTRegressor

from GSAutoML.data_modeling.estimators.modeling_estimator_interface import IModelerEstimator
from GSAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


class TPOTRegressorEstimator(IModelerEstimator):
    def __init__(self, label_col, prediction_col, exclude_cols, random_seed,
                 scoring_metric, time_budget: int):
        super().__init__()
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.scoring_metric = scoring_metric
        self.time_budget = time_budget
        self._seed = random_seed

        self.estimator = TPOTRegressor(scoring=self.scoring_metric,
                                       max_time_mins=self.time_budget,
                                       random_state=self._seed,
                                       )
        self.exclude_cols = exclude_cols

    def fit(self, dataframe, y=None):
        """ fit model on the data and return trained model"""
        print(">>>>>>>>>> In TPOT Regressor estimator >>>>>>>>>>>>>>>>>")
        X = dataframe.copy()
        if self.exclude_cols:
            X.drop(self.exclude_cols, axis=1, inplace=True)
        if self.label_col in list(X.columns):
            y = X[self.label_col]
            X = X.drop(self.label_col, axis=1)
        self.features = X.columns
        self.model = self._get_best_model(X, y)
        return self.model

    def transform(self, X):
        """ add prediction column to the dataframe """
        print(">>>>>>>>>> In TPOT Regressor transformer >>>>>>>>>>>>>>>>>")
        if 'Timestamp' in list(X.columns):
            X.set_index('Timestamp', inplace=True)
        X[self.prediction_col] = self.predict(x=X.drop(columns=[self.label_col], axis=1)) \
            if self.label_col in list(X.columns) else self.predict(x=X)
        return X

    def predict(self, x):
        """ get model predictions """
        return self.model.predict(x)

    def _get_best_model(self, x, y):
        """
        Run the TPOT Regressor to pipeline to find the best model
        :param x: dataframe containing the features to make predictions based on
        :param y: the target col
        :return: best estimator
        """
        self.estimator.fit(features=x, target=y)
        print(self.estimator)
        self.best_pipeline = self.estimator.fitted_pipeline_

        return self.estimator
