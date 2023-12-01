from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class IEstimator(BaseEstimator, TransformerMixin, ABC):
    RENAME = False
    # This class variable is used to mark that this estimator produce new columns that should replace old columns

    def __init__(self):
        super().__init__()
        self.schema_extracted_features = {}  # Artifacts Dictionary for the feature engineering estimators

    @staticmethod
    @abstractmethod
    def accept(pipeline_visitor):
        """ implements the visitor design pattern to build the pipeline stages in runtime according to configurations"""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None) -> TransformerMixin:
        """ Must be implemented to be able to use the class in scikit-learn pipelines"""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Must be implemented to be able to use the class in scikit-learn pipelines"""
        pass

    def get_dataframe(self):
        """Gets the pandas dataframe for debugging"""
        return self.dataframe
