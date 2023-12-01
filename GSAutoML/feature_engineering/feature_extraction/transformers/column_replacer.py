import pandas as pd
from GSAutoML.feature_engineering.common.transformer_interface import ITransformer


class ColumnReplacer(ITransformer):
    def __init__(self,replace_col: str, drop_col: str):
        super().__init__()
        self.replace_col = replace_col
        self.drop_col = drop_col

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_column_replacer()

    def transform(self,df: pd.DataFrame) -> pd.DataFrame:
        dataframe = df.copy()
        dataframe[self.replace_col] = dataframe[self.drop_col]
        dataframe.drop([self.drop_col],axis=1, inplace= True)
        return dataframe
