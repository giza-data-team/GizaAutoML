import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from GSAutoML.enums.classification_algorithms_enum import ClassificationAlgorithmsEnum


class Classifiers:
    """
    A utility class for obtaining classifier initialization and grid search hyperparameters.

    Parameters:
        classifier_name (ClassificationAlgorithmsEnum): The name of the classifier algorithm.
        label_col_name (str): The name of the target label column.
        time_stamp_col_name (str): The name of the timestamp column.
        seasonality_mode (str, optional): The seasonality mode. Default is None.

    Methods:
        get_classifier: Get the initialization of each classifier.
        get_classifier_params: Get grid search hyperparameters for each classifier.
        get_best_classifier(hyperparameters): Get the classifier with specific hyperparameters.

    Attributes:
        classifier_name: The name of the classifier algorithm.
        label_col_name: The name of the target label column.
        time_stamp_col_name: The name of the timestamp column.
        seasonality_mode: The seasonality mode.
    """

    def __init__(self, classifier_name, label_col_name, time_stamp_col_name, seasonality_mode=None):
        self.classifier_name = classifier_name
        self.label_col_name = label_col_name
        self.time_stamp_col_name = time_stamp_col_name
        self.seasonality_mode = seasonality_mode

    def get_classifier(self):
        """ get initialization of each classifier"""
        classifier_algorithm = {
            ClassificationAlgorithmsEnum.AdaboostClassifier: AdaBoostClassifier(),
            ClassificationAlgorithmsEnum.RandomForestClassifier: RandomForestClassifier(),
            ClassificationAlgorithmsEnum.SVC: SVC(),
            ClassificationAlgorithmsEnum.LassoClassifier: linear_model.LogisticRegression(),
            ClassificationAlgorithmsEnum.GaussianProcessClassifier: GaussianProcessClassifier(),
            ClassificationAlgorithmsEnum.XGBoostClassifier: xgb.XGBClassifier(),
            ClassificationAlgorithmsEnum.LightgbmClassifier: lgb.LGBMClassifier(),
            ClassificationAlgorithmsEnum.ElasticNetClassifier: linear_model.LogisticRegression()
        }

        return classifier_algorithm[self.classifier_name]

    def get_classifier_params(self):
        """ get grid search hyperparameters of each classifier """
        # Set a seed for reproducibility
        np.random.seed(42)
        params = {
            ClassificationAlgorithmsEnum.AdaboostClassifier: {'n_estimators': [50, 100, 150, 200, 250],
                                                              'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
                                                              'estimator': [DecisionTreeClassifier(max_depth=1),
                                                                            DecisionTreeClassifier(max_depth=3),
                                                                            DecisionTreeClassifier(max_depth=5)]},
            ClassificationAlgorithmsEnum.RandomForestClassifier: {
                'n_estimators': [100, 200, 250, 400],
                'max_depth': [5, 10, 20, 40]
            },
            ClassificationAlgorithmsEnum.SVC: {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                               'C': [1, 2, 3, 5, 10, 15, 30]},

            ClassificationAlgorithmsEnum.LassoClassifier: {'C': 1 / np.logspace(np.log10(1e-5), np.log10(2), num=30),
                                                           'penalty': ['l1'],
                                                           "solver": ['saga']
                                                           },

            ClassificationAlgorithmsEnum.GaussianProcessClassifier: {
                "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
            },

            ClassificationAlgorithmsEnum.XGBoostClassifier: {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'reg_lambda': [0.2, 0.5, 0.7],
                'gamma': [0.9, 1.16467595, 2.45459974, 2.69871289, 7.65954113, 8.75927882],
                'colsample_bytree': [0.8, 1.0]
            },
            ClassificationAlgorithmsEnum.LightgbmClassifier: {
                'boosting_type': ['gbdt'],
                'num_leaves': [100, 150],
                'learning_rate': [0.001, 0.01],
                'reg_lambda ': [0.01, 0.1, 0.3, 0.5],
                'n_estimators': [50, 300],
                'bagging_freq': [1, 2, 3],
                'max_depth': [3, 5, 9],
                'colsample_bytree': [0.8, 1.0],
                'min_gain_to_split': [0.1, 0.5, 1.0]
            },
            ClassificationAlgorithmsEnum.ElasticNetClassifier: {
                'C': 1 / np.linspace(0.001, .01, 10),
                'l1_ratio': np.linspace(0.0001, 0.001, 10),
                'penalty': ['elasticnet'],
                "solver": ['saga']
            }
        }

        return params[self.classifier_name]

    def get_best_classifier(self, hyperparameters):
        """ get the classifier algorithm initialization with a specific hyperparameters
        :param hyperparameters: dictionary contains the hyperparameters of a specific algorithm
        """
        algorithms = {
            ClassificationAlgorithmsEnum.AdaboostClassifier: lambda: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=hyperparameters.get("estimator", {}).get("max_depth", None)),
                n_estimators=hyperparameters.get("n_estimators", None),
                learning_rate=hyperparameters.get("learning_rate", None),
            ),

            ClassificationAlgorithmsEnum.LassoClassifier: lambda: linear_model.LogisticRegression(
                C=hyperparameters.get("C", None),
                penalty=hyperparameters.get("penalty", None),
                solver=hyperparameters.get("solver", None)
            ),
            ClassificationAlgorithmsEnum.SVC: lambda: SVC(kernel=hyperparameters.get('kernel', None),
                                                          C=hyperparameters.get('C', None)
                                                          ),
            ClassificationAlgorithmsEnum.RandomForestClassifier: lambda: RandomForestClassifier(
                max_depth=hyperparameters.get("max_depth", None),
                n_estimators=hyperparameters.get("n_estimators", None)
            ),
            ClassificationAlgorithmsEnum.GaussianProcessClassifier: lambda: GaussianProcessClassifier(
                kernel=DotProduct(hyperparameters.get("kernel", {}).get("sigma_0", None))
            ),
            ClassificationAlgorithmsEnum.XGBoostClassifier: lambda: xgb.XGBClassifier(
                gamma=hyperparameters.get("gamma", None),
                max_depth=hyperparameters.get("max_depth", None),
                reg_lambda=hyperparameters.get("reg_lambda", None),
                n_estimators=hyperparameters.get("n_estimators", None),
                learning_rate=hyperparameters.get("learning_rate", None),
                colsample_bytree=hyperparameters.get("colsample_bytree", None)
            ),

            ClassificationAlgorithmsEnum.LightgbmClassifier: lambda: lgb.LGBMClassifier(
                max_depth=hyperparameters.get("max_depth", None),
                num_leaves=hyperparameters.get("num_leaves", None),
                reg_lambda=hyperparameters.get("reg_lambda", None),
                n_estimators=hyperparameters.get("n_estimators", None),
                learning_rate=hyperparameters.get("learning_rate", None),
                colsample_bytree=hyperparameters.get("colsample_bytree", None),
                min_gain_to_split=hyperparameters.get("min_gain_to_split", None),
                boosting_type=hyperparameters.get("boosting_type", None),
                bagging_freq=hyperparameters.get("bagging_freq", None)
            ),
            ClassificationAlgorithmsEnum.ElasticNetClassifier: lambda: linear_model.LogisticRegression(
                C=hyperparameters.get("C", None),
                l1_ratio=hyperparameters.get("l1_ratio", None),
                penalty=hyperparameters.get("penalty", None),
                solver=hyperparameters.get("solver", None)
            )

        }

        if self.classifier_name in algorithms:
            return algorithms[self.classifier_name]()
        else:
            raise ValueError("Invalid classifier name")
