from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from pandas import DataFrame, Timestamp

from GSAutoML.enums.aggregations_enums import AggregationsEnum
from GSAutoML.enums.intervals_enum import IntervalEnum


class ResamplerInterface(ABC):
    def __init__(self, date_col="Timestamp", target_col="Target"):
        self.date_col = date_col
        self.target_col = target_col

    @abstractmethod
    def resample_data(self, dataframe: DataFrame, interval: int, time_unit: IntervalEnum,
                      aggregate_func: AggregationsEnum = AggregationsEnum.AVG,
                      multivariate_aggregate_func: dict = None,
                      origin: Union[str, Timestamp] = 'start') -> DataFrame:
        """
        Apply Resampling on the timeseries data
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
        pass

    def _set_date_index(self, dataframe: pd.DataFrame):
        # check if the index is of any the supported types
        if dataframe.index.__class__ not in [pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex]:
            # set the time stamp column to be the index
            dataframe[self.date_col] = pd.to_datetime(dataframe[self.date_col])
            dataframe = dataframe.set_index(self.date_col)
        return dataframe
