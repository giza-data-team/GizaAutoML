import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from GSAutoML.feature_engineering.common.transformer_interface import ITransformer


class MinMaxScalerTransformer(ITransformer):
    """
    A custom transformer for applying Min-Max scaling to a DataFrame while excluding specified columns.

    Parameters:
        excluded_columns (list of str, optional): A list of column names to exclude from scaling.
            Default is an empty list.

    Attributes:
        scaler (MinMaxScaler): The Min-Max scaler object used for scaling the data.
        excluded_columns (list of str): The list of column names to exclude from scaling.

    """

    def __init__(self, excluded_columns=None):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.excluded_columns = excluded_columns if excluded_columns else []

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_normalizer_instance()

    def fit(self, X, y=None):
        """
        Fit the Min-Max scaler on the specified columns of the input DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame to fit the scaler on.
            y (None, optional): Ignored. Included for compatibility.

        Returns:
            self: This instance of the transformer.

        """
        # Exclude specified columns and fit the scaler on the rest of the columns
        columns_to_scale = [col for col in X.columns if col not in self.excluded_columns]
        X_to_scale = X[columns_to_scale]
        self.scaler.fit(X_to_scale)
        return self

    def transform(self, X):
        """
        Apply Min-Max scaling to the specified columns of the input DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame to be scaled.

        Returns:
            pd.DataFrame: A new DataFrame with Min-Max scaled values for specified columns,
                while retaining excluded columns.

        """
        # Exclude specified columns and apply Min-Max scaling to the rest
        columns_to_scale = [col for col in X.columns if col not in self.excluded_columns]
        X_to_scale = X[columns_to_scale]
        scaled_data = self.scaler.transform(X_to_scale)

        # Create a DataFrame with scaled data and the excluded columns
        scaled_df = pd.DataFrame(data=scaled_data, columns=columns_to_scale, index=X.index)
        for col in self.excluded_columns:
            scaled_df[col] = X[col]

        return scaled_df

