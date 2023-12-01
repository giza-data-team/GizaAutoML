from enum import Enum


class AggregationsEnum(Enum):
    """
    Enum for Aggregations functions supported by the Resampler.
    """
    MIN = 'min'
    MAX = 'max'
    SUM = 'sum'
    AVG = 'mean'
    MEAN = 'mean'
    BACKFILL = 'backfill'
    FORWARDFILL = 'ffill'
    INTERPOLATE = 'interpolate'
    MODE = 'mode'
    CATEGORIESCOUNT = 'categories_count'

    def __str__(self):
        return self.value
