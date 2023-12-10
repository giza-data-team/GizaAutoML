from datetime import datetime
from typing import List
from statsmodels.tsa.deterministic import DeterministicProcess
import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.deterministic import CalendarFourier

from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer
import pandas as pd


class SeasonalityFeaturesTransformer(ITransformer):
    def __init__(self, timestamp_col=None, spectrum=None, spectrum_frequency=None, periodogram_power_threshold=None,
                 periodogram_length_threshold=None):
        super().__init__()
        self.peak_frequencies = None
        self.set_timestamp_col_name(timestamp_col)
        self.set_spectrum(spectrum)
        self.set_periodogram_power_threshold(periodogram_power_threshold)
        self.set_periodogram_length_threshold(periodogram_length_threshold)
        self.set_spectrum_freq(spectrum_frequency)
        self.train_time_regressor = None

    def set_timestamp_col_name(self, timestamp_col):
        self.timestamp_col_name = timestamp_col

    def get_timestamp_col_name(self):
        return self.timestamp_col_name

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_spectrum_freq(self, spectrum_freq):
        self.spectrum_freq = spectrum_freq

    def get_spectrum(self):
        return self.spectrum

    def get_spectrum_freq(self):
        return self.spectrum_freq

    def set_periodogram_power_threshold(self, power_threshold):
        self.periodogram_power_threshold = power_threshold

    def get_periodogram_power_threshold(self):
        return self.periodogram_power_threshold

    def set_periodogram_length_threshold(self, length_threshold):
        self.periodogram_length_threshold = length_threshold

    def get_periodogram_length_threshold(self):
        return self.periodogram_length_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, dataframe):
        print(">>>>>>>>>> In seasonality transformer >>>>>>>>>>>>>>>>>")

        df = dataframe.copy()
        if self.timestamp_col_name in df.columns:
            df.set_index(self.timestamp_col_name, inplace=True)
        self.peak_frequencies = self.__get_seasonality_components(self.get_spectrum(), self.get_spectrum_freq())
        ind = df.index.to_numpy() # array of timestamp values
        if self.peak_frequencies:
            spectrum_dict = {"spectrum": [self.get_spectrum_freq().tolist(),
                                          self.get_spectrum().tolist()],
                             "peaks_threshold": self.threshold}

            df_added = self.__get_seasonality_features(self.peak_frequencies, ind)
            #dataframe = dataframe.reset_index()
            all_df = pd.concat([df, df_added], axis=1)
            all_df.reset_index(inplace=True)
            return all_df
        return dataframe

    def __get_seasonality_features(self, peak_frequencies, ind: List[datetime]):
        """
        Generate the seasonality features to merge them with the dataframe.
        :param: spectrum CalendarFourier object containing the seasonality components
        """
        seasonal_features = pd.DataFrame()

        # Convert timestamp column to epoch timestamps
        time_regressor = (ind - np.datetime64('1970-01-01T00:00:00')) // np.timedelta64(1, 's')
        for f in peak_frequencies:
            # Create a mask to retain the dominant frequencies
            seasonal_features[f"cos_{int(1 / f)}"] = np.cos(2 * np.pi * time_regressor * f)
            seasonal_features[f"sin_{int(1 / f)}"] = np.sin(2 * np.pi * time_regressor * f)
        seasonal_features[self.timestamp_col_name] = ind
        seasonal_features = seasonal_features.set_index(self.timestamp_col_name)

        return seasonal_features

    def __get_seasonality_components(self, spectrum: np.ndarray, frequencies: np.ndarray):
        """
        Generate the seasonality features to merge them with the dataframe.
        :param: spectrum
        """
        peaks_inds, _ = find_peaks(spectrum)
        peak_frequencies = []
        self.threshold = np.mean(spectrum[peaks_inds]) + 2 * np.std(spectrum[peaks_inds])
        for index in peaks_inds:
            if spectrum[index] >= self.threshold:
                peak_frequencies.append(frequencies[index])

        return peak_frequencies
