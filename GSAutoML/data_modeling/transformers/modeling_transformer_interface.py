from sklearn.base import TransformerMixin


class IModelerTransformer(TransformerMixin):
    def __init__(self):
        self._seed = 22

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ return trained model"""
        pass
