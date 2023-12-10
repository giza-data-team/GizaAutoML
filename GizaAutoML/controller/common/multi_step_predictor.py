from GizaAutoML.multistep_forecasting.multistep_forecasting import MultistepTimeSeriesForecaster


class MultiStepPredictor:
    """
        A class for performing multi-step time series forecasting and validation.

        This class is designed for multi-step forecasting tasks, where predictions are made for multiple future time steps.
        It encapsulates the necessary data, pipelines, and forecasting components to perform multi-step predictions
        and validate the forecasted results.

        Args:
            train_data: The training dataset used for modeling.
            test_data: The test dataset for making multi-step predictions.
            timestamp_col_name (str): The name of the timestamp column in the datasets.
            prediction_col_name (str): The name of the column where predictions will be stored.
            modeling_pipeline: A pipeline for model training and prediction.
            processing_pipeline: A data processing pipeline for feature engineering and transformation.
            num_lags (int): The number of time lags to consider in the prediction.

        Methods:
            predict():
                Perform multi-step predictions on the test data using the configured pipeline and validate the results.
    """
    def __init__(self, train_data, test_data, timestamp_col_name, prediction_col_name,
                 modeling_pipeline, processing_pipeline,num_lags, target_col):
        self.train_data = train_data
        self.test_data = test_data
        self.target_col = target_col
        self.timestamp_col_name = timestamp_col_name
        self.prediction_col_name = prediction_col_name
        self.fitted_processing_pipeline = processing_pipeline
        self.fitted_modeling_pipeline = modeling_pipeline
        self.significant_lags_no = num_lags

    def predict(self):
        """
            Perform multi-step predictions on the test data using the configured pipeline and validate the results.

            Returns:
                A tuple containing the test data with predictions and validation results.
        """
        num_steps = len(self.test_data)
        print("no of steps:",num_steps)
        print(self.train_data.shape)

        print(self.train_data)

        sampling_freq = self.train_data[self.timestamp_col_name].diff().dt.total_seconds() // 60
        sampling_freq = sampling_freq[1]
        forecaster = MultistepTimeSeriesForecaster(algorithm=self.fitted_modeling_pipeline,
                                                   num_steps=num_steps,
                                                   num_lags=self.significant_lags_no,
                                                   sampling_freq_min=sampling_freq,
                                                   preprocessing_pipeline=self.fitted_processing_pipeline,
                                                   target_col_name=self.target_col,
                                                   timestamp_col_name=self.timestamp_col_name)
        predictions = forecaster.predict(self.train_data)
        test_data_with_predictions = self.test_data.copy()
        test_data_with_predictions[self.prediction_col_name] = predictions
        test_results = forecaster.validate_prediction(predictions, self.test_data)
        return test_data_with_predictions, test_results
