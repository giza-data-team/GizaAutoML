import pandas as pd
# import django
#
# django.setup()
from GSAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer import \
    MinMaxScalerTransformer


class MetaFeaturesExtractor:
    """ A class that has meta features extracting functions. """

    def __init__(self, df: pd.DataFrame, dataset_instance, target_col):
        self._dataset_id = id
        self._dataset = df
        self.target_col = target_col

    def get_num_features(self):
        if 'Timestamp' in self._dataset.columns:
            return len(self._dataset.columns) - 1
        return len(self._dataset.columns)

    def get_num_instances(self):
        return len(self._dataset)

    def get_dataset_ratio(self):
        return self.get_num_instances() / self.get_num_features()

    def get_numerical_to_categorical_ratio(self):
        numerical_features = self._dataset.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self._dataset.select_dtypes(include=['object', 'bool']).columns
        if len(categorical_features) > 0:
            return len(numerical_features) / len(categorical_features)

        else:
            return float("inf")

    def get_num_missing_vals(self):
        return self._dataset.isnull().sum().sum()

    def get_avg_missing_vals_per_feature(self):
        return self.get_num_missing_vals() / self.get_num_features()

    def get_percentage_of_outliers(self):
        Q1 = self._dataset.select_dtypes(include=['int64', 'float64']).quantile(0.25)
        Q3 = self._dataset.select_dtypes(include=['int64', 'float64']).quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self._dataset[(self._dataset.select_dtypes(include=['int64', 'float64'])
                                  < lower_bound) |
                                 (self._dataset.select_dtypes(include=['int64', 'float64'])
                                  > upper_bound)]
        percentage_of_outliers = len(outliers) / self.get_num_instances()
        return percentage_of_outliers

    def get_skewness_features(self):
        skewness = self._dataset.skew(axis=0, numeric_only=True).abs()
        self._skewness_mean = skewness.mean()
        self._skewness_min = skewness.min()
        self._skewness_max = skewness.max()
        self._skewness_std = skewness.std()
        return skewness, self._skewness_mean, self._skewness_min, \
            self._skewness_max, self._skewness_std

    def get_kurtosis_features(self):
        kurtosis = self._dataset.kurtosis(axis=0, numeric_only=True).abs()
        self._kurtosis_mean = kurtosis.mean()
        self._kurtosis_min = kurtosis.min()
        self._kurtosis_max = kurtosis.max()
        self._kurtosis_std = kurtosis.std()
        return kurtosis, self._kurtosis_mean, self._kurtosis_min, \
            self._kurtosis_max, self._kurtosis_std

    def get_sum_symbols(self):
        sum_symbols = self._dataset.select_dtypes(include=['object', 'bool']).nunique().sum()
        return sum_symbols

    def get_avg_symbols(self):
        avg_symbols = self._dataset.select_dtypes(include=['object', 'bool']).nunique().mean()
        return avg_symbols

    def get_std_symbols(self):
        std_symbols = self._dataset.select_dtypes(include=['object', 'bool']).nunique().std()
        return std_symbols

    def get_percentiles_info(self, percentiles_no=10):
        """ calculate statistics of series percentiles"""
        df = self._dataset.sort_values('Timestamp')
        # normalize target column
        normalizer = MinMaxScalerTransformer(excluded_columns=['Timestamp'])
        normalized_df = normalizer.fit_transform(df)
        num_percentiles = percentiles_no
        # Calculate percentile labels while preserving order (division based on indexes)
        normalized_df['Percentile'] = pd.qcut(range(len(normalized_df)), q=num_percentiles, labels=False)
        percentile_stats = normalized_df.groupby('Percentile')[self.target_col].agg(
            ['mean', 'median', 'std', 'min', 'max']).reset_index()

        return percentile_stats
