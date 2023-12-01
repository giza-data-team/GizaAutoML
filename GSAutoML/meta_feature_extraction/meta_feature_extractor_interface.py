from abc import ABC, abstractmethod
import pandas as pd
from GSAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer \
    import MinMaxScalerTransformer


class MetaFeaturesExtractorInterface(ABC):
    """ A class that has meta features extracting functions. """

    def __init__(self, raw_df: pd.DataFrame, imputed_df: pd.DataFrame,
                 target_col: str = 'Target', timestamp_col: str = 'Timestamp'):
        self._dataset_id = id
        self.raw_df = raw_df
        self.imputed_df = imputed_df
        self._target_col = target_col
        self._timestamp_col = timestamp_col

    @abstractmethod
    def extract_meta_features(self) -> dict:
        pass

    def get_num_features(self):
        if self._timestamp_col in self.raw_df.columns:
            return len(self.raw_df.columns) - 1

        return len(self.raw_df.columns)

    def get_dataset_ratio(self):
        return self.get_num_instances() / self.get_num_features()

    def get_num_instances(self):
        return len(self.raw_df)

    def get_missing_count_in_target(self):
        return int(self.raw_df[self._target_col].isnull().sum())

    def get_percentage_of_outliers_in_target(self):
        Q1 = self.imputed_df[self._target_col].quantile(0.25)
        Q3 = self.imputed_df[self._target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        n_outliers = len(self.imputed_df[(self.imputed_df[self._target_col] < lower_bound) |
                                       (self.imputed_df[self._target_col] > upper_bound)])
        percentage_of_outliers = n_outliers / self.get_num_instances()
        return percentage_of_outliers

    def get_target_skewness(self):
        skewness = abs(self.imputed_df[self._target_col].skew())
        return skewness

    def get_target_kurtosis(self):
        kurtosis = abs(self.imputed_df[self._target_col].skew())
        return kurtosis

    def get_percentiles_info(self, percentiles_no=10):
        """ calculate statistics of series percentiles"""
        df = self.imputed_df.sort_values(self._timestamp_col)
        # normalize target column
        normalizer = MinMaxScalerTransformer(excluded_columns=[self._timestamp_col])
        normalized_df = normalizer.fit_transform(df)
        num_percentiles = percentiles_no
        # Calculate percentile labels while preserving order (division based on indexes)
        normalized_df['Percentile'] = pd.qcut(range(len(normalized_df)), q=num_percentiles, labels=False)
        percentile_stats = normalized_df.groupby('Percentile')[self._target_col].agg(
            ['mean', 'median', 'std', 'min', 'max']).reset_index()

        return percentile_stats
