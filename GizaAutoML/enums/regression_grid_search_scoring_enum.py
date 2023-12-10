from enum import Enum


class RegressionScoringMetricEnum(Enum):
    MSE = 'neg_mean_squared_error'
    MAPE= 'neg_mean_absolute_percentage_error'
    MAE = 'neg_mean_absolute_error'
    RMSE = 'neg_root_mean_squared_error'
    R2 = 'r2'
