import pandas as pd
# import django

from GizaAutoML.meta_feature_extraction.meta_feature_extractor_interface import MetaFeaturesExtractorInterface
from GizaAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer \
    import MinMaxScalerTransformer


class MetaFeaturesExtractorMultivariate(MetaFeaturesExtractorInterface):
    """ A class that has meta features extracting functions. """

    def __init__(self, raw_df: pd.DataFrame, imputed_df: pd.DataFrame,
                 target_col: str = 'Target', timestamp_col: str = 'Timestamp'):
        super().__init__(raw_df, imputed_df, target_col, timestamp_col)
        self._predictors_df = imputed_df.drop(columns=[target_col])
        self._numerical_predictors = self._predictors_df.select_dtypes(include=['int64', 'float64']).columns
        self._categorical_predictors = self._predictors_df.select_dtypes(include=['object', 'bool']).columns

    def get_numerical_predictors_count(self):
        return len(self._numerical_predictors)

    def get_categorical_predictors_count(self):
        return len(self._categorical_predictors)

    def get_numerical_to_categorical_ratio(self):
        if len(self._categorical_predictors):
            return self.get_numerical_predictors_count() / self.get_categorical_predictors_count()
        else:
            return None

    def get_numerical_missing_stats(self):
        if len(self._numerical_predictors):
            stat_count, stat_min, stat_max, stat_avg, stat_std = \
                self._get_missing_stats(cols=self._numerical_predictors)
            stats = {'count': int(stat_count),
                     'min': int(stat_min),
                     'max': int(stat_max),
                     'avg': stat_avg,
                     'std': stat_std if stat_std == stat_std else None
                     }
        else:
            stats = self._return_empty_stats()

        return stats

    def get_categorical_missing_stats(self):
        if len(self._categorical_predictors):
            stat_count, stat_min, stat_max, stat_avg, stat_std = \
                self._get_missing_stats(cols=self._categorical_predictors)
            stats = {'count': int(stat_count),
                     'min': int(stat_min),
                     'max': int(stat_max),
                     'avg': stat_avg,
                     'std': stat_std
                     }
        else:
            stats = self._return_empty_stats()

        return stats

    def get_outliers_stats_in_predictors(self):
        if len(self._numerical_predictors):
            Q1 = self._predictors_df[self._numerical_predictors].quantile(0.25)
            Q3 = self._predictors_df[self._numerical_predictors].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            stat_outliers = ((self._predictors_df[self._numerical_predictors] < lower_bound) |
                             (self._predictors_df[self._numerical_predictors] > upper_bound)).sum()

            stat_count = self._predictors_df[self._numerical_predictors].count()   # Count non-empty values

            stat_outliers_perc = stat_outliers / stat_count

            stats_n_fields_with_outliers = (stat_outliers > 0).sum()
            outliers_min = stat_outliers_perc.min()
            outliers_max = stat_outliers_perc.max()
            outliers_avg = stat_outliers_perc.mean()
            outliers_std = stat_outliers_perc.std()

            stats = {'count': int(stats_n_fields_with_outliers),
                     'min': outliers_min,
                     'max': outliers_max,
                     'avg': outliers_avg,
                     'std': outliers_std if outliers_std == outliers_std else None
                     }

        else:
            stats = self._return_empty_stats()

        return stats

    def get_predictors_skewness(self):
        if len(self._numerical_predictors):
            skewness = self._predictors_df[self._numerical_predictors].skew(axis=0).abs()
            skewness_std = skewness.std()
            stats = {'min': skewness.min(),
                     'max': skewness.max(),
                     'avg': skewness.mean(),
                     'std': skewness_std if skewness_std == skewness_std else None
                     }
        else:
            stats = self._return_empty_stats()

        return stats

    def get_predictors_kurtosis(self):
        if len(self._numerical_predictors):
            kurtosis = self._predictors_df[self._numerical_predictors].kurtosis(axis=0).abs()
            kurtosis_std = kurtosis.std()
            stats = {'min': kurtosis.min(),
                     'max': kurtosis.max(),
                     'avg': kurtosis.mean(),
                     'std': kurtosis_std if kurtosis_std == kurtosis_std else None
                     }
        else:
            stats = self._return_empty_stats()
        return stats

    def get_categorical_predictors_stats(self):
        if len(self._categorical_predictors):
            categories = self._predictors_df[self._categorical_predictors].nunique()
            stats = {'min': int(categories.min()),
                     'max': int(categories.max()),
                     'avg': categories.mean(),
                     'std': categories.std()
                     }
        else:
            stats = self._return_empty_stats()
        return stats

    def get_percentiles_info(self, percentiles_no=10):
        """ calculate statistics of series percentiles"""
        df = self.imputed_df.sort_values(self._timestamp_col)
        df = df[[self._target_col]]
        # normalize target column
        normalizer = MinMaxScalerTransformer()
        normalized_df = normalizer.fit_transform(df)
        # Calculate percentile labels while preserving order (division based on indexes)
        normalized_df['Percentile'] = pd.qcut(range(len(normalized_df)), q=percentiles_no, labels=False)
        percentile_stats = normalized_df.groupby('Percentile')[self._target_col] \
                                        .agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        return percentile_stats

    def extract_meta_features(self) -> dict:
        num_missing_stats = self.get_numerical_missing_stats()
        cat_missing_stats = self.get_categorical_missing_stats()
        predictors_outliers_stats = self.get_outliers_stats_in_predictors()
        skewness_stats = self.get_predictors_skewness()
        kurtosis_stats = self.get_predictors_kurtosis()
        categorical_stats = self.get_categorical_predictors_stats()

        meta_features = {'features_no': self.get_num_features(),
                         'instances_no': self.get_num_instances(),
                         'dataset_ratio': self.get_dataset_ratio(),
                         'missing_count_in_target': self.get_missing_count_in_target(),
                         'target_skewness': self.get_target_skewness(),
                         'target_kurtosis': self.get_target_kurtosis(),
                         'perc_target_outliers': self.get_percentage_of_outliers_in_target(),
                         'count_numerical_vars': self.get_numerical_predictors_count(),
                         'count_categorical_vars': self.get_categorical_predictors_count(),
                         'numerical_to_categorical_ratio': self.get_numerical_to_categorical_ratio(),
                         'num_vars_n_nulls': num_missing_stats['count'],
                         'num_vars_min_nulls': num_missing_stats['min'],
                         'num_vars_max_nulls': num_missing_stats['max'],
                         'num_vars_avg_nulls': num_missing_stats['avg'],
                         'num_vars_std_nulls': num_missing_stats['std'],
                         'cat_vars_n_nulls': cat_missing_stats['count'],
                         'cat_vars_min_nulls': cat_missing_stats['min'],
                         'cat_vars_max_nulls': cat_missing_stats['max'],
                         'cat_vars_avg_nulls': cat_missing_stats['avg'],
                         'cat_vars_std_nulls': cat_missing_stats['std'],
                         'num_predictors_outliers_count': predictors_outliers_stats['count'],
                         'num_predictors_outliers_min_perc': predictors_outliers_stats['min'],
                         'num_predictors_outliers_max_perc': predictors_outliers_stats['max'],
                         'num_predictors_outliers_avg_perc': predictors_outliers_stats['avg'],
                         'num_predictors_outliers_std_perc': predictors_outliers_stats['std'],
                         'min_num_predictors_skewness': skewness_stats['min'],
                         'max_num_predictors_skewness': skewness_stats['max'],
                         'avg_num_predictors_skewness': skewness_stats['avg'],
                         'std_num_predictors_skewness': skewness_stats['std'],
                         'min_num_predictors_kurtosis': kurtosis_stats['min'],
                         'max_num_predictors_kurtosis': kurtosis_stats['max'],
                         'avg_num_predictors_kurtosis': kurtosis_stats['avg'],
                         'std_num_predictors_kurtosis': kurtosis_stats['std'],
                         'min_categories': categorical_stats['min'],
                         'max_categories': categorical_stats['max'],
                         'avg_categories': categorical_stats['avg'],
                         'std_categories': categorical_stats['std'],
                         }
        return meta_features

    def _get_missing_stats(self, cols):
        null_stats = self._predictors_df[cols].isnull().sum()
        stats_n_fields_with_nulls = (null_stats > 0).sum()
        stats_min = null_stats.min()
        stats_max = null_stats.max()
        stats_avg = null_stats.mean()
        stats_std = null_stats.std()

        return stats_n_fields_with_nulls, stats_min, stats_max, stats_avg, stats_std

    @staticmethod
    def _return_empty_stats():
        stats = {'count': None,
                 'min': None,
                 'max': None,
                 'avg': None,
                 'std': None
                 }
        return stats
