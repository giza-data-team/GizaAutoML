import pandas as pd
from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer


class TargetShifter(ITransformer):
    def __init__(self,target_col: str, n_lags:int):
        super().__init__()
        self.target_col = target_col
        self.n_lags = n_lags

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_shifter_stage()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shifts the values in the specified target column by the given number of lags.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            target_col (str): The name of the target column to be shifted.
            n_lags (int): The number of lags by which to shift the target column.
                          A positive value indicates a forward shift,
                          while a negative value indicates a backward shift.

        Returns:
            pd.DataFrame: A new DataFrame with the target column shifted by the specified number of lags.

        Raises:
            ValueError: If the absolute value of n_lags exceeds the length of the DataFrame.

        Example:
            To shift a target column 'y' by 3 lags:
            >>> shifted_df = TargetShifter.transform(df, 'y', 3)
        """

        dataframe = df.copy()
        if abs(self.n_lags) >= dataframe.shape[0]:
            raise ValueError('Number of lags cannot exceed dataframe length')

        if self.n_lags > 0:
            dataframe['Target_original'] = dataframe[self.target_col]

        dataframe[self.target_col] = dataframe[self.target_col].shift(self.n_lags)

        if self.n_lags >= 0:
            return dataframe.iloc[self.n_lags:]

        else:
            return dataframe.iloc[:self.n_lags]
