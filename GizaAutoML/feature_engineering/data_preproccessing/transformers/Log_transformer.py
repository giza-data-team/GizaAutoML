import pandas as pd
import numpy as np
from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer
from GizaAutoML.enums.time_series_types_enum import TimeSeriesTypesEnum

class LogTransformer(ITransformer):
    """
    A custom transformer for applying natural logarithm transformation to a DataFrame column.

    Parameters:
        column_name (str): The name of the column to transform.

    Attributes:
        column_name (str): The name of the column to transform.
    """

    def __init__(self, column_name):
        super().__init__()
        self.column_name = column_name

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_log_transformer_instance()

    def fit(self, X, y=None):
        """
        Fit the transformer (no action needed for this transformer).

        Parameters:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Ignored. Included for compatibility.

        Returns:
            self: This instance of the transformer.
        """
        return self

    def transform(self, X):
        """
        Apply natural logarithm transformation to the specified column of the input DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame with logarithmically transformed column replacing the original one.
        """
        # Check if the column exists in the DataFrame
        if self.column_name not in X.columns:
            raise ValueError(f"Column '{self.column_name}' not found in the DataFrame.")

        X_transformed = X.copy()
        min_val= min(X_transformed[self.column_name])
        print(min_val)
        print(abs(min_val+1))
        #shift the data to the positive side if column contains -ve values
        if min_val <= 0:
            X_transformed[self.column_name] = X_transformed[self.column_name]+(abs(min_val)+1)

        print(X_transformed[self.column_name])
        X_transformed[self.column_name] = np.log(X_transformed[self.column_name])
        X_transformed.dropna(inplace=True)

        return X_transformed



# # Sample DataFrame (replace this with your own data)
# data = {
#     'Value': [-100, 110, 121, -133.1, 146.41, 161.051, 177.1561],
#     'Other_Column': [1, 2, 3, 4, 5, 6, 7]
# }
#
# df = pd.DataFrame(data)
#
# # Instantiate the LogTransformer for the 'Value' column with seasonality_mode
# transformed_df = LogTransformer(column_name='Value').transform(df)
#
# # Display the original and transformed DataFrame
# print("Original DataFrame:")
# print(df)
#
# print("\nTransformed DataFrame:")
# print(transformed_df)