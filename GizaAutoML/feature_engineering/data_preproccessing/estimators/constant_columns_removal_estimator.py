from GizaAutoML.feature_engineering.common.estimator_interface import IEstimator
from GizaAutoML.feature_engineering.data_preproccessing.transformers.constant_columns_removal_transformer import \
    ConstantColumnsRemovalTransformer
import pandas as pd


class ConstantColumnsRemovalEstimator(IEstimator):
    """ ِA custom Estimator for applying constant columns removal"""

    def __init__(self):
        super(ConstantColumnsRemovalEstimator, self).__init__()
        self.constant_columns = []

    @staticmethod
    def _is_constant_column(df, col_name):
        """
        return whether the column is constant or not.
        The column consider constant if it's empty, or have Only one value,
        or have number of values equal to number of instances

        :param df (pd.DataFrame): dataframe to get constant columns in.
        :param col_name (str): column name to check whether it's constant or not

        Returns: True if column is constant and False if not
        """
        n_instances = df.shape[0]
        n_unique = df[col_name].nunique()
        if (n_unique <= 1) or (n_unique == n_instances and df[col_name].dtype == 'object'):
            return True
        return False

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_consant_columns_removal()

    def fit(self,  X, y=None):
        """ get the constant columns """
        print(">>>>>>>>>> In constant columns Removal estimator >>>>>>>>>>>>>>>>>")
        df = X.copy()
        for column in df.columns:
            if self._is_constant_column(df, column):
                self.constant_columns.append(column)
        print(self.constant_columns)
        return self

    def transform(self, X: pd.DataFrame):
        """ Apply ConstantColumnsRemovalTransformer to remove constant columns """
        self.transformer = ConstantColumnsRemovalTransformer(self.constant_columns)
        return self.transformer.transform(X)
