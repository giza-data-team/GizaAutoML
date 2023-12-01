import pandas as pd
import numpy as np
from feature_engine.creation import CyclicalFeatures
from GSAutoML.feature_engineering.common.transformer_interface import ITransformer
from math import pi, sin, cos

class CyclicalEncoderTransformer(ITransformer):
    def __init__(self, columns_to_encode):
        super().__init__()
        self.set_columns_to_encode(columns_to_encode)

    def set_columns_to_encode(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode

    def get_columns_to_encode(self):
        return self.columns_to_encode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        # Iterate over columns to check if the max is zero to add epsilon to avoid division by zero
        for column in X.columns:
            # Check if the maximum value in the column is zero
            if X[column].max() == 0:
                # Add 1 to all values in the column
                X[column] += 1
            print(X[column])
        print(">>>>>>>>>> In cyclical encoder transformer >>>>>>>>>>>>>>>>>")
        cyclical_transformer = CyclicalFeatures(variables=self.get_columns_to_encode(), drop_original=True)
        X = cyclical_transformer.fit_transform(X)
        return X

# # Example usage:
# df = pd.DataFrame({
#     'day': [-1,0,-0.5,-0.4],
#     'months': [1, 12,6,3],
# })
#
# # Create an instance of the CyclicalEncoderTransformer
# cyclical_encoder = CyclicalEncoderTransformer(columns_to_encode=['day', 'months'])
#
# # Apply the transformation to the DataFrame
# df_transformed = cyclical_encoder.transform(df)
# print(df_transformed)