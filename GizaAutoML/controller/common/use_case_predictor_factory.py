from GizaAutoML.controller.common.single_step_predictor import SingleStepPredictor
from GizaAutoML.controller.common.multi_step_predictor import MultiStepPredictor


class PredictorFactory:
    """
       A factory class for creating predictive models, specifically designed for time series forecasting tasks.

       This class is responsible for creating instances of either a SingleStepPredictor or a MultiStepPredictor,
       depending on the value of the `multi_step_flag`. It encapsulates the configuration parameters and pipelines
       needed for training and evaluation of predictive models.

       Args:
           multi_step_flag (bool): A flag indicating whether to create a single-step or multi-step predictor.
           utils: A utility object for various helper functions.
           evaluator: An evaluator object for model evaluation.
           evaluation_metric_enum: An enumeration of evaluation metrics to be used.
           timestamp_col_name (str): The name of the timestamp column in the dataset (for multi-step predictor).
           target_col_name (str): The name of the target column in the dataset.
           prediction_col_name (str): The name of the prediction column in the dataset.
           processing_pipeline: A pipeline for data preprocessing.
           modeling_pipeline: A pipeline for model training and prediction.

       Methods:
           create_predictor(train_data, test_data, num_lags):
               Create and return a predictor instance based on the configuration and flag.
    """
    def __init__(self, multi_step_flag, utils, evaluator, evaluation_metric_enum,
                 timestamp_col_name, target_col_name, prediction_col_name,
                 processing_pipeline, modeling_pipeline):
        self.multi_step_flag = multi_step_flag
        self.utils = utils
        self.evaluation_metric_enum = evaluation_metric_enum
        self.target_col_name = target_col_name
        self.timestamp_col_name = timestamp_col_name
        self.prediction_col_name = prediction_col_name
        self.evaluator = evaluator
        self.processing_pipeline = processing_pipeline
        self.modeling_pipeline = modeling_pipeline

    def create_predictor(self, train_data, test_data, num_lags):
        """
        Create and return a predictor instance based on the configuration and flag.

        Args:
            train_data: The training dataset.
            test_data: The test dataset.
            num_lags (int): The number of time lags to consider in the prediction.

        Returns:
            A SingleStepPredictor instance if multi_step_flag is False, or
            a MultiStepPredictor instance if multi_step_flag is True.
        """
        if not self.multi_step_flag:
            return SingleStepPredictor(train_data=train_data,
                                       test_data=test_data,
                                       target_col_name=self.target_col_name,
                                       prediction_col_name=self.prediction_col_name,
                                       processing_pipeline=self.processing_pipeline,
                                       modeling_pipeline=self.modeling_pipeline,
                                       utils=self.utils,
                                       num_lags=num_lags,
                                       evaluator=self.evaluator,
                                       evaluation_metric_enum=self.evaluation_metric_enum)
        else:
            return MultiStepPredictor(train_data=train_data,
                                      test_data=test_data,
                                      timestamp_col_name=self.timestamp_col_name,
                                      prediction_col_name=self.prediction_col_name,
                                      modeling_pipeline=self.modeling_pipeline,
                                      processing_pipeline=self.processing_pipeline,
                                      num_lags=num_lags,
                                      target_col=self.target_col_name)
