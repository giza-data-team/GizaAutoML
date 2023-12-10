from GizaAutoML.data_modeling.estimators.modeling_estimator_interface import IModelerEstimator
from sklearn.model_selection import GridSearchCV
from GizaAutoML.enums.classification_evaluation_metric_enum import ClassificationEvaluationMetricEnum
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from GizaAutoML.data_modeling.classifiers import Classifiers
from sklearn.metrics import make_scorer


class ClassificationEstimator(IModelerEstimator):
    """
    A class for performing classification model training and prediction.

    Args:
        label_col (str): The name of the target label column in the dataset.
        prediction_col (str): The name of the prediction column to be added to the dataset.
        params (dict): Hyperparameters for the classifier.
        estimator: The classification estimator.
        scoring_metric (ClassificationScoringMetricEnum): The scoring metric for model evaluation.
        exclude_cols (list, optional): Columns to exclude from the dataset during training. Default is None.
        hyperparameters (dict, optional): Additional hyperparameters for grid search. Default is None.
        algorithm_name (str, optional): Name of the classification algorithm. Default is None.
    Attributes:
        features: A list of feature column names.
        model: The trained classification model.
        time_stamp_col_name: The name of the timestamp column.
        best_params: The best hyperparameters selected during training.
        cv_folds_no: Number of cross-validation folds. Default is 5.
        grid_search_results: Results of the grid search.
    Methods:
        get_model_initialization: Get the classification algorithm initialization with specific hyperparameters.
        fit(dataframe, y=None): Fit the model on the data and return the trained model.
        transform(X): Add prediction column to the dataframe.
        predict(x): Get model predictions.
        features_importance(): Get feature importance for the best model.
        _get_best_model(x, y): Apply grid search on estimator to get the best hyperparameters.

    """

    def __init__(self, label_col, prediction_col, time_stamp_col_name, params, estimator,
                 scoring_metric: ClassificationEvaluationMetricEnum, exclude_cols=None, hyperparameters=None,
                 algorithm_name=None):
        super().__init__()
        self.features = None
        self.model = None
        self.time_stamp_col_name = time_stamp_col_name
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.scoring_metric = scoring_metric
        self.params = params
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.algorithm_name = algorithm_name
        self.cv_folds_no = 5
        self.exclude_cols = exclude_cols
        self.grid_search_results = None
        self.best_params = None

    def get_model_initialization(self):
        """get the classification algorithm initialization with a specific hyperparameters"""

        classifiers = Classifiers(classifier_name=self.algorithm_name, label_col_name=self.label_col,
                                  time_stamp_col_name=self.time_stamp_col_name)
        # get model instance initialized with hyperparameters
        estimator = classifiers.get_best_classifier(hyperparameters=self.hyperparameters)
        self.best_params = self.hyperparameters
        return estimator

    def fit(self, dataframe, y=None):
        """ fit model on the data and return trained model"""
        print(">>>>>>>>>> In classifier estimator >>>>>>>>>>>>>>>>>")

        X = dataframe.copy()
        if self.exclude_cols:
            X.drop(self.exclude_cols, axis=1, inplace=True)
        if 'Timestamp' in list(X.columns):
            X.set_index('Timestamp', inplace=True)
        if self.label_col in list(X.columns):
            y = X[self.label_col]
            X = X.drop(self.label_col, axis=1)
        self.features = X.columns
        if self.hyperparameters:
            self.model = self.get_model_initialization()
            self.model.fit(X, y)
        else:
            self.model = self._get_best_model(X, y)
        return self.model

    def transform(self, X):
        """ add prediction column to the dataframe """
        print(">>>>>>>>>> In classifier transformer >>>>>>>>>>>>>>>>>")

        if 'Timestamp' in list(X.columns):
            X.set_index('Timestamp', inplace=True)
        X[self.prediction_col] = self.predict(x=X.drop(columns=[self.label_col], axis=1)) \
            if self.label_col in list(X.columns) else self.predict(x=X)
        return X

    def predict(self, x):
        """ get model predictions """
        return self.model.predict(x)

    def features_importance(self):
        """ get feature importance for the best model"""

        importance_df = pd.DataFrame({'Feature': self.features,
                                      'Importance': self.model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df

    def _get_best_model(self, x, y):
        """
        apply grid search on estimator to get the best hyperparameters
        :param x: dataframe containing the features to make predictions based on
        :param y: the target col
        :return: best estimator
        """
        score = make_scorer(ClassificationEvaluationMetricEnum[self.scoring_metric.name].value.func,
                            **ClassificationEvaluationMetricEnum[self.scoring_metric.name].value.keywords)
        grid_search = GridSearchCV(estimator=self.estimator,
                                   param_grid=self.params,
                                   cv=TimeSeriesSplit(n_splits=self.cv_folds_no).split(x, y),
                                   scoring=score,
                                   return_train_score=True,
                                   n_jobs=1)  # set n_jobs to 1 to ensure deterministic results
        # fit grid search
        grid_search.fit(x, y)

        cv_result_df = pd.DataFrame(grid_search.cv_results_).set_index('rank_test_score').sort_index()
        self.grid_search_results = cv_result_df[['mean_fit_time', 'std_fit_time', 'mean_score_time',
                                                 'std_score_time', 'params', 'mean_test_score',
                                                 'std_test_score', 'mean_train_score', 'std_train_score']]
        self.best_params = grid_search.best_params_

        return grid_search.best_estimator_
