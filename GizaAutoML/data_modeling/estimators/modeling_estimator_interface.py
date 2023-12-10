from sklearn.base import BaseEstimator, RegressorMixin


class IModelerEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self._seed = 22

    def fit(self, X, y=None):
        """ return trained model"""
        pass

    def transform(self, X):
        """ Must be implemented to be able to use the class in scikit-learn pipelines"""
        pass
