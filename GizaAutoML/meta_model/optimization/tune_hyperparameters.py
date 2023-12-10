import pandas as pd
import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from sklearn import datasets, svm
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.metrics import make_scorer, mean_absolute_error
from smac.initial_design import DefaultInitialDesign
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
from GizaAutoML.enums.algorithms_mapping import AlgorithmsMappingEnum
from sklearn.gaussian_process.kernels import DotProduct


class HyperparameterOptimizerBO:
    def __init__(self, space, X, y, random_seed,algorithm_name=None, n_trials=100):
        self.space = space
        self.algorithm_name = algorithm_name
        self.n_trials = n_trials
        self.X =X
        self.y=y
        self.random_seed = random_seed

    def train(self, config, seed: int ) -> float:
        config_dict = dict(config)
        np.random.seed(self.random_seed)

        # Create a scoring function for MAE
        # todo: change to support different scoring metrics
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=True)

        if self.algorithm_name == AlgorithmsMappingEnum.AdaboostRegressor.value:
            regressor = self.algorithm_name(**config_dict,
                                            estimator=DecisionTreeRegressor(max_depth=5))#todo: change to get from defaults
        if self.algorithm_name == AlgorithmsMappingEnum.GaussianProcessRegressor.value:
            # todo: handle different kernels
            regressor = self.algorithm_name(kernel=DotProduct(sigma_0=0.1)
                                            ,**config_dict)

        else:
            regressor = self.algorithm_name(**config_dict)
        tscv = TimeSeriesSplit(n_splits=3)
        # Use cross_val_score with the MAE scoring function
        scores = cross_val_score(regressor, self.X, self.y, cv=5, scoring=mae_scorer)

        cost = np.mean(scores)

        print("MAE cost:", cost)
        return cost
