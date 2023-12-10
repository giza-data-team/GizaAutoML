from autosklearn.regression import AutoSklearnRegressor
from GizaAutoML.data_modeling.estimators.modeling_estimator_interface import IModelerEstimator
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
# from Trainer.enums.evaluation_metric_enum import RegressionEvaluationMetricEnum
from GizaAutoML.enums.autosklearn_evaluation_metric import AutoSklearnEvaluationMetricEnum
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import get_scorer
from autosklearn.metrics import make_scorer
import pandas as pd


class AutoSKLearnRegressorEstimator(IModelerEstimator):
    def __init__(self, label_col, prediction_col, exclude_cols,random_seed,
                 scoring_metric: AutoSklearnEvaluationMetricEnum, time_budget: int):
        super().__init__()
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.scoring_metric = scoring_metric
        self.time_budget = time_budget
        self._seed = random_seed
        # self.estimator = AutoSklearnRegressor(time_left_for_this_task=self.time_budget*60,
        #                                       metric=make_scorer(self.scoring_metric.name,
        #                                                          get_scorer(self.scoring_metric.value)._score_func))
        self.estimator = AutoSklearnRegressor(time_left_for_this_task=self.time_budget * 60,
                                              metric=scoring_metric.value,
                                              seed=self._seed,
                                              resampling_strategy='cv',
                                              resampling_strategy_arguments={"train_size": 0.8,
                                                                             "shuffle": False,
                                                                             'folds': 5},
                                              ensemble_size=0
                                              )
        self.exclude_cols = exclude_cols

    def fit(self, dataframe, y=None):
        """ fit model on the data and return trained model"""
        print(">>>>>>>>>> In Scikit Learn Regressor estimator >>>>>>>>>>>>>>>>>")
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
        print(">>>>>>>>>> In AutoSKLearn Regressor transformer >>>>>>>>>>>>>>>>>")
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
        Run the AutoSklearn Regressor to pipeline to find the best model
        :param x: dataframe containing the features to make predictions based on
        :param y: the target col
        :return: best estimator
        """
        self.estimator = self.estimator.fit(x, y)
        # TODO: try to extract the algorithm of the best model along with the hyperparameters
        # print(self.estimator.show_models())
        return self.estimator
