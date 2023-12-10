from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline


class MachineLearningPipeline(ABC):
    """
    Abstract base class for creating machine learning pipelines for both regression and classification tasks.

    Args:
        label_col (str): Name of the label column in the dataset.
        prediction_col (str): Name of the column where predictions will be stored.
        exclude_cols (list): List of column names to exclude from the feature set.
        scoring_metric (Enum): The evaluation metric to be used for model performance.
        estimator (class): The name of the estimator or classifier to be used.
        time_stamp_col_name (str, optional): Name of the timestamp column in the dataset.
        seasonality_mode (str, optional): Seasonality mode if applicable.
        hyperparameters (dict, optional): Dictionary containing hyperparameters for the estimator.


    Methods:
        create_estimator: Abstract method to create and return the estimator object specific to the task.
        _get_estimator: Abstract method to get the estimator object.
        _get_estimator_params_grid: Abstract method to get estimator parameters for grid search.
        _get_stages: Abstract method to get pipeline stages.
        fit(X, y, **kwargs): Fits the machine learning pipeline to the training data.
        transform(X): Transforms the input data using the trained pipeline.
    """
    def __init__(self, label_col, prediction_col, exclude_cols, scoring_metric, estimator, time_stamp_col_name=None,
                 seasonality_mode=None, hyperparameters=None):
        super().__init__()
        self.pipeline = None
        self.label_col_name = label_col
        self.prediction_col_name = prediction_col
        self.scoring_metric = scoring_metric
        self.time_stamp_col_name = time_stamp_col_name
        self.seasonality_mode = seasonality_mode
        self.hyperparameters = hyperparameters
        self.exclude_cols = exclude_cols
        self.estimator_name = estimator
        self.estimator = self.create_estimator()
        self.steps = self._get_stages()

    @abstractmethod
    def create_estimator(self):
        """ Create and return the estimator object specific to the task (regressor or classifier). """
        pass

    @abstractmethod
    def _get_estimator(self):
        """ get regressor object """
        pass

    @abstractmethod
    def _get_estimator_params_gird(self):
        """ get estimator params for grid search """
        pass

    @abstractmethod
    def _get_stages(self):
        """ get pipeline stages """
        pass

    def fit(self, X, y=None, **kwargs):
        pipeline = Pipeline(steps=self.steps)
        self.pipeline = pipeline.fit(X=X, y=y)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return X


