from GizaAutoML.enums.aggregations_enums import AggregationsEnum
from GizaAutoML.enums.intervals_enum import IntervalEnum
import pandas as pd
from GizaAutoML.data_resampler.resampler import Resampler


class DataConstructor:
    def __init__(self, date_col, target_col):
        self.date_col = date_col
        self.target_col = target_col
        self.resampler = Resampler(date_col=self.date_col, target_col=self.target_col)

    def resample(self, dataframe: pd.DataFrame, agg_func) -> pd.DataFrame:
        """
        Resample dataframe according to forecasting rate
        :return: resampled pandas dataframe
        """
        df = dataframe.copy()
        if self.date_col in list(df.columns):
            df.sort_values(self.date_col, inplace=True)
            df.set_index(self.date_col, inplace=True)

        freq_counts = df.index.to_series().diff().value_counts()
        # Sort the freq_counts Series from highest to lowest
        freq_counts = freq_counts.sort_values(ascending=False)

        freq, time_unit = self._get_major_forecasting_rate(freq_counts)

        pd_resampled_data = self.resampler.resample_data(df,
                                                         interval=freq,
                                                         time_unit=time_unit,
                                                         aggregate_func=agg_func)
        dataframe = pd_resampled_data.reset_index()
        return dataframe

    def _get_major_forecasting_rate(self, freq_counts):
        # major freq will always be the first value as we sort the values Desc
        major_freq_minutes = pd.Timedelta(freq_counts.index.values[0]).total_seconds()//60

        if major_freq_minutes < 1:
            major_freq_seconds = int(pd.Timedelta(freq_counts.index.values[0]).total_seconds())
            print("major freq in seconds", major_freq_seconds)

            return major_freq_seconds, IntervalEnum.SECOND
        else:
            print("major freq in minutes", major_freq_minutes)

            return major_freq_minutes, IntervalEnum.MINUTE
