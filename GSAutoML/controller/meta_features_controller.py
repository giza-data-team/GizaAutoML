import pandas as pd

from GSAutoML.controller.common.utils import Utils
from GSAutoML.meta_feature_extraction.meta_feature_extractor_univariate import MetaFeaturesExtractorUnivariate
from GSAutoML.meta_feature_extraction.meta_feature_extractor_multivariate import MetaFeaturesExtractorMultivariate
from GSAutoML.meta_feature_extraction.time_series_features_extractor import TimeSeriesFeaturesExtractor
from GSAutoML.feature_engineering.feature_extraction.estimators.trend_feature_estimator import TrendFeatureEstimator
from GSAutoML.feature_engineering.data_preproccessing.estimators.series_type_estimator import SeriesTypeEstimator
from GSAutoML.feature_engineering.data_preproccessing.estimators.prophet_imputer_estimator import ProphetImputerEstimator
from GSAutoML.controller.common.knowledge_base_collector import KnowledgeBaseCollector


class MetaFeaturesController:
    """ A class that has meta features extracting functions. """

    def __init__(self, dataframe: pd.DataFrame, dataset_instance, target_col="Target",
                 timestamp_col="Timestamp", save_results=True):
        self.dataset_instance = dataset_instance  # todo: no need for if saving will be with end points
        if dataset_instance:
            if isinstance(dataset_instance, dict):
                self.dataset_instance_name = dataset_instance['name']
                self.dataset_instance_id = dataset_instance['id']
            else:
                self.dataset_instance_name = dataset_instance.name
                self.dataset_instance_id = dataset_instance.id
        else:
            self.dataset_instance_name = None
            self.dataset_instance_id = None

        self._target_col_name = target_col
        self._timestamp_col_name = timestamp_col
        self.utils = Utils(series_col=self._target_col_name,
                           timestamp_col_name=self._timestamp_col_name)

        self.raw_df = dataframe.copy()  # resampled df
        self._series_type = SeriesTypeEstimator().series_type_estimator(dataframe=self.raw_df.dropna())

        # select numerical columns to impute
        columns_to_impute = self.raw_df.select_dtypes(include=['int64', 'float64']).columns
        imputer = ProphetImputerEstimator(timestamp_col=self._timestamp_col_name,
                                          input_cols=[self._target_col_name]
                                          if self._check_if_univariate()
                                          else columns_to_impute,
                                          seasonality_mode=self._series_type)
        self.imputed_df = imputer.fit_transform(self.raw_df)
        self.KB_collector = KnowledgeBaseCollector()
        self._meta_features_extractor = self._create_metafeature_extractor()
        self.percentiles_df = self._meta_features_extractor.get_percentiles_info()
        self.time_series_extractor = TimeSeriesFeaturesExtractor(self.imputed_df,
                                                                 date_col=self._timestamp_col_name)
        self._stationary_feature, self._non_stationary_feature, self._stationary_feature_1_dif, \
            self._stationary_feature_2_dif = self.time_series_extractor.get_stationary_feature(
            col_name=self._target_col_name)
        self._trend_type = TrendFeatureEstimator(target_col_name=self._target_col_name,
                                                 timestamp_col=timestamp_col,
                                                 seasonality_mode=self._series_type).get_trend_type(
            self.imputed_df[self._target_col_name])
        self.meta_features = self._get_results()
        print(self.meta_features)
        if save_results:
            self.save_results()

    def _get_results(self):
        pacf_values = self.time_series_extractor.get_first_n_pacf_values(col_name=self._target_col_name)
        meta_features = self._meta_features_extractor.extract_meta_features()

        meta_features.update({'dataset_name': self.dataset_instance_name,
                              'sampling_rate': self.time_series_extractor.get_sampling_rate(),
                              'stationary_no': self._stationary_feature,
                              'non_stationary_no': self._non_stationary_feature,
                              'first_diff_stationary_no': self._stationary_feature_1_dif,
                              'second_diff_stationary_no': self._stationary_feature_2_dif,
                              'lags_no': self.time_series_extractor.get_lagged_features(col_name=self._target_col_name),
                              'insignificant_lags_no': self.time_series_extractor.get_insignificant_lags(
                                  col_name=self._target_col_name),
                              'seasonality_components_no': self.time_series_extractor.get_seasonality_components(
                                  self._series_type, col_name=self._target_col_name),
                              'fractal_dim': self.time_series_extractor.get_fractal_dimension(
                                  col_name=self._target_col_name),
                              'series_type': self._series_type[self._target_col_name],
                              'trend_type': self._trend_type})

        for i in range(1, len(pacf_values) + 1):
            if i != 10:
                meta_features[f"pacf_0{i}"] = pacf_values[i - 1]
            else:
                meta_features[f"pacf_{i}"] = pacf_values[i - 1]

        for i in range(len(self.percentiles_df)):
            meta_features[f'percentile_{i}_min'] = \
                self.percentiles_df[self.percentiles_df['Percentile'] == i]['min'].iloc[0]
            meta_features[f'percentile_{i}_max'] = \
                self.percentiles_df[self.percentiles_df['Percentile'] == i]['max'].iloc[0]
            meta_features[f'percentile_{i}_mean'] = \
                self.percentiles_df[self.percentiles_df['Percentile'] == i]['mean'].iloc[0]
            meta_features[f'percentile_{i}_median'] = \
                self.percentiles_df[self.percentiles_df['Percentile'] == i]['median'].iloc[0]
            meta_features[f'percentile_{i}_std'] = \
                self.percentiles_df[self.percentiles_df['Percentile'] == i]['std'].iloc[0]

        return meta_features

    def save_results(self):
        if self._check_if_univariate():
            return self._save_univariate_results()

        return self._save_multivariate_results()

    def _save_univariate_results(self):
        meta_features = self.meta_features.copy()
        meta_features['dataset'] = self.dataset_instance_id
        meta_features.pop('dataset_name')
        self.KB_collector.add_uni_variate_meta_features(meta_features)

    def _save_multivariate_results(self):
        meta_features = self.meta_features
        meta_features['dataset'] = self.dataset_instance_id
        meta_features.pop('dataset_name')
        self.KB_collector.add_multi_variate_meta_features(meta_features)

    def _check_if_univariate(self) -> bool:
        """
        Check if the DataFrame is univariate, meaning it contains only one variable other than the datetime column.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        bool: True if the DataFrame is univariate, False otherwise.
        """
        columns = self.raw_df.columns
        print(columns)
        if len(columns) == 1:
            return True

        if len(columns) > 2:
            return False

        return set(columns) == {self._timestamp_col_name, self._target_col_name}

    def _create_metafeature_extractor(self):
        if self._check_if_univariate():
            return MetaFeaturesExtractorUnivariate(self.raw_df,
                                                   self.imputed_df,
                                                   self._target_col_name,
                                                   self._timestamp_col_name)

        return MetaFeaturesExtractorMultivariate(self.raw_df,
                                                 self.imputed_df,
                                                 self._target_col_name,
                                                 self._timestamp_col_name)
