from GSAutoML.feature_engineering.common.transformer_interface import ITransformer


class ConstantColumnsRemovalTransformer(ITransformer):
    """ A custom transformer to remove constant columns from a dataframe """
    def __init__(self, constant_columns):
        super(ConstantColumnsRemovalTransformer, self).__init__()
        self.constant_columns = constant_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ remove constant columns from a df """
        if self.constant_columns:
            X.drop(self.constant_columns, axis=1, inplace=True)
        return X

