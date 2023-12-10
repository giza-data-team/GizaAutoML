from GizaAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor


class SingleStepPredictor:
    """
        A class for performing single-step time series predictions and evaluating the model.

        This class is designed for forecasting tasks where predictions are made one time step ahead. It encapsulates
        the necessary data, pipelines, and evaluation components to predict and assess the model's performance.

        Args:
            train_data: The training dataset used for modeling.
            test_data: The test dataset for making predictions.
            target_col_name (str): The name of the target column in the datasets.
            prediction_col_name (str): The name of the column where predictions will be stored.
            processing_pipeline: A data processing pipeline for feature engineering and transformation.
            modeling_pipeline: A pipeline for model training and prediction.
            utils: A utility object for various helper functions.
            num_lags (int): The number of time lags to consider in the prediction.
            evaluator: An evaluator object for model evaluation.
            evaluation_metric_enum: An enumeration of evaluation metrics to be used.

        Methods:
            predict():
                Perform predictions on the test data using the configured pipeline and evaluate the model.
    """
    def __init__(self, train_data, test_data, target_col_name, prediction_col_name,
                 processing_pipeline, modeling_pipeline,
                 utils, num_lags, evaluator, evaluation_metric_enum):
        self.train_data = train_data
        self.test_data = test_data
        self.target_col_name = target_col_name
        self.prediction_col_name = prediction_col_name
        self.fitted_processing_pipeline = processing_pipeline
        self.fitted_modeling_pipeline = modeling_pipeline
        self.significant_lags_no = num_lags
        self.utils = utils
        self.evaluation_metric_enum = evaluation_metric_enum
        self.evaluator = evaluator

    def predict(self):
        """
            Perform predictions using the configured pipeline and evaluate the model.

            Returns:
                A tuple containing the test data with predictions and the evaluation results.
        """
        parent_pipeline = self.utils.create_pipeline([self.fitted_processing_pipeline, self.fitted_modeling_pipeline])

        # concat lags in test data
        test_lagged_processor = LaggedDataPreprocessor(train_data=self.train_data, test_data=self.test_data,
                                                       num_lags=self.significant_lags_no)
        self.test_data = test_lagged_processor.preprocess_data()

        # transforming on test data
        test_data_with_predictions = parent_pipeline.transform(self.test_data)
        print("......... Evaluate the pipeline on test data .............")

        test_results = self.evaluator.evaluate(evaluation_metric_enum=self.evaluation_metric_enum,
                                               actual_data=test_data_with_predictions[self.target_col_name],
                                               predicted_data=test_data_with_predictions[self.prediction_col_name])
        return test_data_with_predictions, test_results

