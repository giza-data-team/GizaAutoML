from GizaAutoML.enums.classification_evaluation_metric_enum import ClassificationEvaluationMetricEnum
from GizaAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GizaAutoML.enums.ML_tasks_enum import MLTasksEnum
from enum import Enum


class Evaluator:
    """
    A class that encapsulates evaluation metrics, allowing them to be interchangeable based on the metric type.

    Parameters:
        metric_type (MLTasksEnum): The type of machine learning task, either classification or regression.

    Methods:
        evaluate(actual_data, predicted_data, evaluation_metric_enum=None):
            Evaluates the given metrics on actual and predicted data.

    Attributes:
        metrics: The set of available evaluation metrics based on the specified metric_type.
    """

    def __init__(self, metric_type):
        """
        Initialize the Evaluator.

            Parameters:
            metric_type (MLTasksEnum): The type of machine learning task, either classification or regression.
        """
        if metric_type == MLTasksEnum.CLASSIFICATION:
            self.metrics = ClassificationEvaluationMetricEnum
        elif metric_type == MLTasksEnum.REGRESSION:
            self.metrics = RegressionEvaluationMetricEnum
        else:
            raise ValueError("Invalid metric type")

    def evaluate(self, actual_data, predicted_data, evaluation_metric_enum=None):
        """
        Evaluate the specified evaluation metrics on actual and predicted data.

        Parameters:
            actual_data (array-like): The actual target values.
            predicted_data (array-like): The predicted target values.
            evaluation_metric_enum (Union[Enum, List[Enum]], optional):
                A single evaluation metric or a list of metrics to evaluate.
                If None, all available metrics for the specified task type will be evaluated.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        evaluation_metrics_list = []  # Create a list of the metrics
        if isinstance(evaluation_metric_enum, Enum):  # This means it is a single metric
            evaluation_metrics_list.append(evaluation_metric_enum)
        else:
            evaluation_metrics_list = evaluation_metric_enum
        all_results = {}
        if evaluation_metric_enum is None:
            evaluation_metrics = self.metrics
        for current_evaluation_metric in evaluation_metrics_list:
            try:
                # loop over enums and return a dictionary to controller
                metric_function = getattr(self.metrics, current_evaluation_metric.name).value
                result = metric_function(actual_data, predicted_data)
                all_results[current_evaluation_metric.name] = result
            except AttributeError:
                raise ValueError(f"Metric '{current_evaluation_metric}' not found")
        return all_results
