import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

from GizaAutoML.enums.algorithms_mapping import AlgorithmsMappingEnum
from GizaAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GizaAutoML.enums.statistical_models_enum import StatisticalModelsEnum


class Regressors:
    """
    A utility class for obtaining classifier initialization and grid search hyperparameters.

    Parameters:
        regressor_name (RegressionAlgorithmsEnum): The name of the regression algorithm.
        label_col_name (str): The name of the target label column.
        time_stamp_col_name (str): The name of the timestamp column.
        seasonality_mode (str, optional): The seasonality mode. Default is None.

    Methods:
        get_classifier: Get the initialization of each classifier.
        get_classifier_params: Get grid search hyperparameters for each classifier.
        get_best_classifier(hyperparameters): Get the classifier with specific hyperparameters.

    Attributes:
        regressor_name: The name of the regression algorithm.
        label_col_name: The name of the target label column.
        time_stamp_col_name: The name of the timestamp column.
        seasonality_mode: The seasonality mode.
    """

    def __init__(self, regressor_name, label_col_name, time_stamp_col_name, seasonality_mode=None):
        self.regressor_name = regressor_name
        self.label_col_name = label_col_name
        self.time_stamp_col_name = time_stamp_col_name
        self.seasonality_mode = seasonality_mode

    def get_regressor(self):
        """ get initialization of each regressor"""
        regression_algorithms = {
            RegressionAlgorithmsEnum.AdaboostRegressor: AdaBoostRegressor(),
            RegressionAlgorithmsEnum.RandomForestRegressor: RandomForestRegressor(),
            RegressionAlgorithmsEnum.SVR: SVR(),
            RegressionAlgorithmsEnum.LassoRegressor: linear_model.LassoCV(),
            RegressionAlgorithmsEnum.GaussianProcessRegressor: GaussianProcessRegressor(),
            RegressionAlgorithmsEnum.XGBoostRegressor: xgb.XGBRegressor(),
            RegressionAlgorithmsEnum.LightgbmRegressor: lgb.LGBMRegressor(),
            RegressionAlgorithmsEnum.ElasticNetRegressor: linear_model.ElasticNetCV(),
            RegressionAlgorithmsEnum.ExtraTreesRegressor: ExtraTreesRegressor()
        }

        return regression_algorithms[self.regressor_name]

    def get_regressor_params(self, smac_optimization=False, random_seed=42):
        """ get grid search hyperparameters of each regressor """
        # Set a seed for reproducibility
        np.random.seed(random_seed)
        Gaussian_kernels = [(RBF(length_scale=1.0), "RBF"),
                            (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)), "RBF with Constant"),
                            (DotProduct(sigma_0=0.1), "DotProduct")]

        params = {
            RegressionAlgorithmsEnum.AdaboostRegressor: {'n_estimators': [10, 100, 150, 200],
                                                         'learning_rate': [0.01, 0.1, 0.5, 1.0],
                                                         'loss': ['linear', 'square', 'exponential'],
                                                         'estimator': [DecisionTreeRegressor(max_depth=3),
                                                                       DecisionTreeRegressor(max_depth=5)]
                                                         },
            RegressionAlgorithmsEnum.RandomForestRegressor: {
                'n_estimators': [100, 200, 250, 400],
                'max_depth': [3, 5, 10, 20]
            },
            RegressionAlgorithmsEnum.SVR: {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                           'C': [1, 2, 3, 5, 10],
                                           'epsilon': [0.01, 0.05, 0.1]},
            RegressionAlgorithmsEnum.LassoRegressor: {'alpha': np.logspace(np.log10(1e-5), np.log10(2), num=30)},

            RegressionAlgorithmsEnum.GaussianProcessRegressor: {
                "alpha": [1e-2, 1e-3, 1e-4],
                "kernel": [kernel for kernel, _ in Gaussian_kernels]
            },

            RegressionAlgorithmsEnum.XGBoostRegressor: {
                'n_estimators': [20, 200],
                'max_depth': [2, 7],
                'learning_rate': [0.1, 1],
                'reg_lambda': [0.8, 10],
                'gamma': [0.9, 1.16467595, 2.248149123539492, 3.9963209507789],
                'colsample_bytree': [0.5, 1.0],
                'subsample': [0.1, 1]
            },
            RegressionAlgorithmsEnum.LightgbmRegressor: {
                'boosting_type': ['gbdt'],
                'num_leaves': [100, 150],
                'learning_rate': [0.05, 0.5],
                'reg_lambda ': [0, 100],
                'n_estimators': [10, 100, 150, 200],
                'bagging_freq': [1, 2],
                'max_depth': [3, 5],
                'colsample_bytree': [0.8, 1.0],
                'min_gain_to_split': [0.1, 0.5],
                'reg_alpha': [0, 100]
            },
            RegressionAlgorithmsEnum.ElasticNetRegressor: {
                # 'alpha': np.linspace(0.1, 1, 10),
                'l1_ratio': np.linspace(0.3, 1, 10)
            },
            RegressionAlgorithmsEnum.ExtraTreesRegressor: {'n_estimators': [50, 512],
                                                           'max_features': [0.2, 1],
                                                           'min_samples_split': [5, 10, 20],
                                                           'min_samples_leaf': [4, 12],
                                                           'bootstrap': [True],
                                                           'criterion': ['friedman_mse'],
                                                           'warm_start': [True]
                                                           },
            StatisticalModelsEnum.Prophet: {'changepoint_prior_scale': [0.01, 0.1, 0.5],
                                            'seasonality_prior_scale': [0.01, 0.1, 1.0],
                                            'holidays_prior_scale': [0.01, 0.1, 1.0]}
        }
        if smac_optimization:
            params[RegressionAlgorithmsEnum.AdaboostRegressor].pop('estimator')
            params[RegressionAlgorithmsEnum.GaussianProcessRegressor].pop('kernel')

        return params[self.regressor_name]

    def get_best_regressor(self, hyperparameters):
        """ get the regression algorithm initialization with a specific hyperparameters
        :param hyperparameters: dictionary contains the hyperparameters of a specific algorithm
        """

        return AlgorithmsMappingEnum[self.regressor_name.name].value(**hyperparameters)
