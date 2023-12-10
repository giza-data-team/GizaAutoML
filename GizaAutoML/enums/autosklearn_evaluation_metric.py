from enum import Enum
from autosklearn.metrics import mean_squared_error, mean_absolute_error


class AutoSklearnEvaluationMetricEnum(Enum):
    # regression evaluation metrics
    # the enum members here are the metric functions, so we could use them directly
    MSE = mean_squared_error
    MAE = mean_absolute_error

