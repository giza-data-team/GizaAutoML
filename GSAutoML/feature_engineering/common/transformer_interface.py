from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class ITransformer(BaseEstimator, TransformerMixin, ABC):
    def __init__(self):
        super().__init__()

    # @abstractmethod
    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        """ Must be implemented to be able to use the class as a stage in scikit-learn pipeline"""
        pass

    def get_dataframe(self):
        return self.dataframe

    def drop_and_replace(self, X, output_cols, input_cols):
        X = X.drop(input_cols, axis=1)
        X.rename(columns={old: new for old, new in zip(output_cols, input_cols)}, inplace=True)
        return X
