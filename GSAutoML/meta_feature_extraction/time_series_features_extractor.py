import pandas as pd
import numpy as np
from scipy.signal import periodogram
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from GSAutoML.feature_engineering.feature_extraction.estimators.seasonality_features_estimator import SeasonalityFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.lagged_features_estimator import LaggedFeaturesEstimator


class TimeSeriesFeaturesExtractor:
    def __init__(self, dataframe: pd.DataFrame, date_col):
        self.dataframe = dataframe
        self._stationary_feature = 0
        self._stationary_feature_1_dif = 0
        self._stationary_feature_2_dif = 0
        self._non_stationary_feature = 0
        self._sampling_period = float()
        self.n_lags = 30
        self.n_cons_non_significant_lags = 3
        self._lagged_estimator = None
        self.date_col = date_col

    def get_stationary_feature(self, col_name="Target"):
        numeric_features = self.dataframe.select_dtypes(include=['int64', 'float64'])

        if bool(adfuller(self.dataframe[col_name])[1] < 0.05):
            self._stationary_feature += 1
        else:
            self._non_stationary_feature += 1
            df = pd.DataFrame(
                {'stationary_feature_1': self.dataframe[col_name].diff()})
            # drop nans that exist after differencing
            df = df.dropna()
            if bool(adfuller(df)[1] < 0.05):
                self._stationary_feature_1_dif += 1
            else:
                df = pd.DataFrame({'stationary_feature_2': df['stationary_feature_1'].diff()})
                df = df.dropna()
                if bool(adfuller(df)[1] < 0.05):
                    self._stationary_feature_2_dif += 1
        return self._stationary_feature, self._non_stationary_feature, \
            self._stationary_feature_1_dif, self._stationary_feature_2_dif

    def get_sampling_rate(self):
        # Sampling Frequency (Mins)
        X = self.dataframe.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])

        time_diff = X[self.date_col].diff()
        self._sampling_period = time_diff.dt.total_seconds() / 60
        return self._sampling_period.iloc[1]

    def get_seasonality_components(self, series_type,col_name="Target"):
        df = self.dataframe.reset_index()
        seasonality_estimator = SeasonalityFeaturesEstimator(time_series_type=series_type,
                                                             target_col_name=col_name,
                                                             timestamp_col=self.date_col)
        seasonality_estimator.fit(df)
        seasonality_estimator.transform(df)
        peak_freqs = seasonality_estimator.get_peak_frequencies()
        if peak_freqs:
            return len(peak_freqs)
        return 0

    def __generate_periodogram(self, ts: np.ndarray, fs: float) -> tuple:
        """
        Compute the periodogram values.
        :param ts: Numpy Array like Time series of measurement values.
        :param fs: Float of the sampling frequency.
        :return: Tuple of numpy arrays of the frequencies and the power of these frequencies.
        """
        f, spectrum = periodogram(
            ts, fs=fs, detrend=False,
            window='boxcar',
            scaling='spectrum')
        return f, spectrum

    def get_hurst_exponent(self, col_name="Target", max_lags_threshold=10):
        """Returns the Hurst Exponent of the time series"""
        lags = range(2, max_lags_threshold)

        # variances of the lagged differences
        tau = [np.std(np.subtract(self.dataframe[col_name][lag:],
                                  self.dataframe[col_name][:-lag])) for lag in lags]

        # Check if there are any non-zero lag values
        if np.count_nonzero(tau) == 0:
            return 0.0  # or any other default value or handling

        # calculate the slope of the log plot -> the Hurst Exponent
        non_zero_indices = np.nonzero(tau)
        non_zero_lags = np.array(lags)[non_zero_indices]
        non_zero_tau = np.array(tau)[non_zero_indices]
        reg = np.polyfit(np.log(non_zero_lags), np.log(non_zero_tau), 1)

        return reg[0]

    def get_fractal_dimension(self, col_name="Target", k_max=4):
        """
        Calculate the Higuchi Fractal Dimension of a 1-D time series.
        k_max (int): The maximum value of k (number of subdivisions).
        """
        L = np.zeros((k_max,))
        x_len = len(self.dataframe[col_name])

        # Calculate the length of each subseries
        for k in range(1, k_max + 1):
            Lk = np.zeros((k,))
            for m in range(k):
                idx = np.arange(m, x_len, k)
                Lmk = np.sum(np.abs(np.diff(self.dataframe[col_name][idx])))
                Lmk = (Lmk * (x_len - 1) / ((x_len - 1) // k)) / k
                Lk[m] = Lmk.mean()

            L[k - 1] = np.log(np.mean(Lk))

        # Fit a line to log(Lk) vs log(1/k)
        coeffs = np.polyfit(np.log(np.arange(1, k_max + 1)), L, 1)
        return coeffs[0]

    def get_lagged_features(self, col_name="Target"):
        """
        get number of significant lags of a feature using PACF
        """
        if not self._lagged_estimator:
            self._lagged_estimator = LaggedFeaturesEstimator(cols=[col_name])
            self._lagged_estimator.fit(self.dataframe)
        return self._lagged_estimator.col_lags_dic[col_name]['significant_lags']

    def get_insignificant_lags(self, col_name="Target"):
        """
        get number of insignificant lags between the significant lags
        """
        if not self._lagged_estimator:
            self._lagged_estimator = LaggedFeaturesEstimator(cols=[col_name])
            self._lagged_estimator.fit(self.dataframe)
        return self._lagged_estimator.col_lags_dic[col_name]['non_significant_lags_count']

    def get_first_n_pacf_values(self, col_name='Target', n_lags=10):
        """
        returns a list containing the first n_lags PACF Values
        """
        if not self._lagged_estimator:
            self._lagged_estimator = LaggedFeaturesEstimator(cols=[col_name])
            self._lagged_estimator.fit(self.dataframe)
        return self._lagged_estimator.pacf_result[col_name][1:n_lags + 1]
