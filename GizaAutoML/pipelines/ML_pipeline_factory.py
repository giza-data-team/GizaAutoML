from GizaAutoML.pipelines.regression_pipeline import RegressorPipeline
from GizaAutoML.pipelines.classification_pipeline import ClassificationPipeline
from GizaAutoML.enums.ML_tasks_enum import MLTasksEnum


class MLFactory:
    """
    A factory class to create machine learning pipelines for classification and regression tasks.

    Methods:
        create_pipeline(task_type, label_col, prediction_col, exclude_cols, scoring_metric, estimator,
                        time_stamp_col_name=None, seasonality_mode=None, hyperparameters=None):
            Create and return a machine learning pipeline based on the specified task type.

    """
    @staticmethod
    def create_pipeline(task_type, label_col, prediction_col, exclude_cols, scoring_metric, estimator,
                        time_stamp_col_name=None, seasonality_mode=None, hyperparameters=None):
        if task_type == MLTasksEnum.CLASSIFICATION:
            return ClassificationPipeline(label_col, prediction_col, exclude_cols, scoring_metric, estimator,
                                          time_stamp_col_name, seasonality_mode, hyperparameters
                                          )
        elif task_type == MLTasksEnum.REGRESSION:
            return RegressorPipeline(label_col, prediction_col, exclude_cols, scoring_metric, estimator,
                                     time_stamp_col_name, seasonality_mode, hyperparameters
                                     )
        else:
            raise ValueError("Invalid task_type. Supported values: 'CLASSIFICATION' or 'REGRESSION'")
