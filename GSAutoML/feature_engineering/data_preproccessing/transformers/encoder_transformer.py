from GSAutoML.feature_engineering.common.transformer_interface import ITransformer
import pandas as pd


class EncoderTransformer(ITransformer):

    def __init__(self, columns_to_encode, encoders):
        super(EncoderTransformer, self).__init__()
        self.columns_to_encode = columns_to_encode
        self.label_encoders = encoders

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Transforming dataframe by encoding categorical columns """
        if self.columns_to_encode:
            X_encoded = X.copy()

            for column, le in self.label_encoders.items():

                X_encoded[column] = self.label_encoders[column].transform(X_encoded[column])+1

            return X_encoded
        else:
            return X
