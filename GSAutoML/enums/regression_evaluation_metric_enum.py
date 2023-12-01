from enum import Enum
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from functools import partial  # So we can use the functions names and values in the evaluator class


class RegressionEvaluationMetricEnum(Enum):
    # classification evaluation metrics
    # the enum members here are the metric functions, so we could use them directly
    MSE = partial(mean_squared_error)
    MAPE = partial(mean_absolute_percentage_error)
    MAE = partial(mean_absolute_error)
    R2 = partial(r2_score)
