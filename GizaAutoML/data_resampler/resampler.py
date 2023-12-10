from typing import Union

from pandas import DataFrame, Timestamp

from GizaAutoML.enums.aggregations_enums import AggregationsEnum
from GizaAutoML.enums.intervals_enum import IntervalEnum
from GizaAutoML.data_resampler.resampler_interface import ResamplerInterface
from GizaAutoML.data_resampler.univariate_resampler import UnivariateResampler
from GizaAutoML.data_resampler.multivariate_resampler import MultivariateResampler


class Resampler:
    def __init__(self, date_col="Timestamp", target_col="Target"):
        self.date_col = date_col
        self.target_col = target_col

    def resample_data(self, dataframe: DataFrame, interval: int, time_unit: IntervalEnum,
                      aggregate_func: AggregationsEnum = AggregationsEnum.AVG,
                      multivariate_aggregate_func: dict = None,
                      origin: Union[str, Timestamp] = 'start') -> DataFrame:
        """
        Resamples input dataframe using a suitable resampler object.
        :param:
            dataframe: Pandas dataframe to be resampled.
            interval: the resampling frequency
            time_unit: the unit of the time by which you want to resample the data,
                       such as: Min for minute, H for hour, D for Day.
            aggregate_func: the function used for aggregation and resampling (For target variable).
            multivariate_aggregate_func: dictionary for  aggregation functions to be used with multivariate datasets.
                                         In case of univariate, this parameter is ignored.
                                         In case of multivariate, only variables in the dictionary keys are aggregated,
                                         other variables are ignored.
                                         Ex. {field1: AggregationsEnum.MIN, field2: AggregationsEnum.MAX,
                                              field3: AggregationsEnum.AVG}
            origin (str or pd.Timestamp, default 'start'): The timestamp on which to adjust the grouping.
                If string, must be one of the following:
                - 'epoch': origin is 1970-01-01
                - 'start': origin is the first value of the timeseries
                - 'start_day': origin is the first day at midnight of the timeseries
                - 'end': origin is the last value of the timeseries
                - 'end_day': origin is the ceiling midnight of the last day

        :return:
            resampling_df: the dataframe containing the data after resampling
        """
        resampler = self._create_resampler(dataframe)
        return resampler.resample_data(dataframe, interval, time_unit, aggregate_func,
                                       multivariate_aggregate_func, origin)

    def _create_resampler(self, dataframe: DataFrame) -> ResamplerInterface:
        """
        Factory method that checks the input dataframe and creates a suitable Resampler object accordingly.
        Args:
            dataframe: The pandas DataFrame to be resampled.

        Returns:
            A resampler object that matches the input dataframe.
        """
        if self._check_if_univariate(dataframe):
            return UnivariateResampler(date_col=self.date_col, target_col=self.target_col)

        return MultivariateResampler(date_col=self.date_col, target_col=self.target_col)

    def _check_if_univariate(self, df: DataFrame) -> bool:
        """
        Check if the DataFrame is univariate, meaning it contains only one variable other than the datetime column.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        bool: True if the DataFrame is univariate, False otherwise.
        """
        columns = df.columns
        if len(columns) == 1:
            return True

        if len(columns) > 2:
            return False

        return set(columns) == {self.date_col, self.target_col}
