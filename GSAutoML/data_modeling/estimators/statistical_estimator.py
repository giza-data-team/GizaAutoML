from GSAutoML.data_modeling.estimators.modeling_estimator_interface import IModelerEstimator
from sklearn.model_selection import GridSearchCV
from GSAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


class StatisticalEstimator(IModelerEstimator):
    def __init__(self, label_col, prediction_col, params, estimator, scoring_metric: RegressionScoringMetricEnum):
        super().__init__()
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.scoring_metric = scoring_metric
        self.params = params
        self.estimator = estimator
        self.cv_folds_no = 5

    def fit(self, dataframe, y=None):
        """ fit model on the data and return trained model"""
        print(">>>>>>>>>> In regression estimator >>>>>>>>>>>>>>>>>")
        self.estimator.params = self.params
        self.model, self.grid_search_results, self.best_params = self.estimator.fit(dataframe)
        return self.model

    def transform(self, X):
        """ add prediction column to the dataframe """
        print(">>>>>>>>>> In regressor transformer >>>>>>>>>>>>>>>>>")
        self.estimator.transform(X)
        return X



