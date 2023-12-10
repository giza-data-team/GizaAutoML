import numpy as np
import pandas as pd
from scipy.signal import periodogram

from GizaAutoML.feature_engineering.common.estimator_interface import IEstimator
from GizaAutoML.feature_engineering.feature_extraction.transformers.seasonality_features_transformer import \
    SeasonalityFeaturesTransformer
from GizaAutoML.feature_engineering.feature_extraction.estimators.trend_feature_estimator import \
    TrendFeatureEstimator
from GizaAutoML.enums.time_series_types_enum import TimeSeriesTypesEnum
from GizaAutoML.feature_engineering.data_preproccessing.transformers.Log_transformer import LogTransformer


class SeasonalityFeaturesEstimator(IEstimator):

    def __init__(self, time_series_type, target_col_name="Target", timestamp_col= "Timestamp"):
        super().__init__()
        self.time_series_type = time_series_type
        self.spectrum = None
        self.target_col_name = target_col_name
        self.timestamp_col = timestamp_col
        self.periodogram_power_threshold = 2.0
        self.periodogram_length_threshold = 30
        self.periodogram_detrend = "linear"
        self.periodogram_window = "boxcar"
        self.periodogram_scaling = "spectrum"

    def get_spectrum_freq(self):
        return self.freq

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_seasonality_features_instance()

    def fit(self, df: pd.DataFrame, y=None):
        print(">>>>>>>>>> In seasonality estimator >>>>>>>>>>>>>>>>>")
        dataframe = df.copy()
        time_series_type = self.time_series_type[self.target_col_name]

        if "Trend" not in dataframe.columns: # if The trend stage is not in the pipeline
            trend_extractor = TrendFeatureEstimator(seasonality_mode=time_series_type,
                                                    target_col_name=self.target_col_name,
                                                    timestamp_col=self.timestamp_col
                                                    )
            trend_extractor.fit(dataframe)

            trend = trend_extractor.transform(dataframe)['Trend']
        else:
            trend = dataframe["Trend"]


        # according to type of the time series substract or divide the trend component to make data stationary
        if time_series_type == TimeSeriesTypesEnum.ADDITIVE.value:
            dataframe[self.target_col_name] = dataframe[self.target_col_name] - trend
        elif time_series_type == TimeSeriesTypesEnum.MULTIPLICATIVE.value:
            dataframe[self.target_col_name] = dataframe[self.target_col_name] / trend

        #If the series type is multiplicative apply log transformation first before generating the periodogram
        if time_series_type == TimeSeriesTypesEnum.MULTIPLICATIVE.value:
            df_transformed_target = LogTransformer(column_name=self.target_col_name).transform(dataframe)
            freq, spectrum = self.__generate_periodogram(df_transformed_target[self.target_col_name].to_numpy(), 1)
        elif time_series_type == TimeSeriesTypesEnum.ADDITIVE.value:
            freq, spectrum = self.__generate_periodogram(dataframe[self.target_col_name].to_numpy(), 1)

        self.spectrum = spectrum
        self.freq = freq
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transformer = SeasonalityFeaturesTransformer(
            timestamp_col=self.timestamp_col, spectrum=self.spectrum,
            spectrum_frequency=self.freq,
            periodogram_power_threshold=self.periodogram_power_threshold,
            periodogram_length_threshold=self.periodogram_length_threshold)
        transformed_df = self.transformer.transform(X)
        return transformed_df

    def get_peak_frequencies(self):
        return self.transformer.peak_frequencies

    def __generate_periodogram(self, ts: np.ndarray, fs: float) -> tuple:
        """
        Compute the periodogram values.
        :param ts: Numpy Array like Time series of measurement values.
        :param fs: Float of the sampling frequency.
        :return: Tuple of numpy arrays of the frequencies and the power of these frequencies.
        """

        f, spectrum = periodogram(
            ts, fs=fs, detrend=False,
            window=self.periodogram_window,
            scaling=self.periodogram_scaling)
        return f, spectrum
