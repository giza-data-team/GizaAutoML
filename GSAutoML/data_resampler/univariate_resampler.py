import pandas as pd

from GSAutoML.data_resampler.resampler_interface import ResamplerInterface
from GSAutoML.enums.aggregations_enums import AggregationsEnum
from GSAutoML.enums.intervals_enum import IntervalEnum


class UnivariateResampler(ResamplerInterface):
    def __init__(self, date_col="Timestamp", target_col="Target"):
        super().__init__(date_col, target_col)

    def resample_data(self, dataframe: pd.DataFrame, interval: int, time_unit: IntervalEnum,
                      aggregate_func: AggregationsEnum = AggregationsEnum.AVG,
                      multivariate_aggregate_func: dict = None,
                      origin: str = 'start'):

        # apply Averaging as the default aggregation function
        aggregate_func = aggregate_func.value

        # check if the index is of any the supported types
        dataframe = self._set_date_index(dataframe)

        resampling_df = dataframe.resample(str(interval) + time_unit.value, origin=origin).agg(aggregate_func)
        return resampling_df
