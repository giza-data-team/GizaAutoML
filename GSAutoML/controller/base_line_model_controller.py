import pandas as pd

from GSAutoML.enums.ML_tasks_enum import MLTasksEnum
from GSAutoML.data_modeling.evaluator import Evaluator
from GSAutoML.data_resampler.data_constructor import DataConstructor
from GSAutoML.enums.aggregations_enums import AggregationsEnum
from GSAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GSAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GSAutoML.split_data.split_data import TimeSeriesSplitter


class BaseModelController:
    """
        A controller class for training and evaluating a base model that makes predictions based
         only on the first lag.

        Args:
            dataframe (pd.DataFrame): The input time series dataset.
            dataset_name (str): The name of the dataset.
            results_path (str): The path to save training and test results.

        Attributes:
            dataframe (pd.DataFrame): The input time series dataset.
            timestamp_col_name (str): The name of the timestamp column in the dataset.
            target_col_name (str): The name of the target variable column in the dataset.
            prediction_col_name (str): The name of the column for model predictions.
            dataset_name (str): The name of the dataset.
            results_path (str): The path to save training and test results.
        """

    def __init__(self, dataframe, dataset_name, results_path):
        self.dataframe = dataframe
        self.timestamp_col_name = 'Timestamp'
        self.target_col_name = 'Target'
        self.prediction_col_name = 'prediction'
        self.dataset_name = dataset_name
        self.results_path = results_path

    def _resample_real_data(self, df):
        """
        Apply resampling on the dataset.

        Args:
            df (pd.DataFrame): The input time series dataset.

        Returns:
            pd.DataFrame: The resampled time series dataset.
        """
        date_col = self.timestamp_col_name
        target_col = self.target_col_name
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df.rename(columns={df.columns[0]: target_col})
        df.reset_index(inplace=True)
        # sort values by date_col
        df.sort_values(by=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)
        # resample dataset
        constructor = DataConstructor(date_col=self.timestamp_col_name, target_col=target_col)
        resampled_df = constructor.resample(dataframe=df, agg_func=AggregationsEnum.AVG)
        return resampled_df

    def _predict_with_first_lag(self, df):
        """
           Predict target values with a lag of one time step.
           Args:
               df (pd.DataFrame): The input time series dataset.

           Returns:
               pd.DataFrame: The input dataset with predictions for the next time step.
        """
        # Shift the values by one time step to create the next prediction
        df[self.prediction_col_name] = df[self.target_col_name].shift(1)
        df = df.dropna()

        return df

    def train(self):
        """
            Train the base model, evaluate its performance, and save results.

            Returns:
                dict: A dictionary containing training and evaluation results.
        """
        resampled_df = self._resample_real_data(self.dataframe)
        # split data
        train_data, test_data = (TimeSeriesSplitter(value_column=self.target_col_name,
                                                    timestamp_column=self.timestamp_col_name)
                                 .split_data(data=resampled_df))

        # concat lags in test data
        test_lagged_processor = LaggedDataPreprocessor(train_data=train_data,
                                                       test_data=test_data,
                                                       num_lags=1)
        test_data = test_lagged_processor.preprocess_data()

        train_data_with_predictions = self._predict_with_first_lag(train_data)

        test_data_with_predictions = self._predict_with_first_lag(test_data)

        # evaluate model performance on training data
        evaluator = Evaluator(MLTasksEnum.REGRESSION)
        train_MAPE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAPE,
                                        actual_data=train_data_with_predictions[self.target_col_name],
                                        predicted_data=train_data_with_predictions[self.prediction_col_name])

        train_MSE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MSE,
                                       actual_data=train_data_with_predictions[self.target_col_name],
                                       predicted_data=train_data_with_predictions[self.prediction_col_name])

        train_MAE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAE,
                                       actual_data=train_data_with_predictions[self.target_col_name],
                                       predicted_data=train_data_with_predictions[self.prediction_col_name])
        train_r2 = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.R2,
                                      actual_data=train_data_with_predictions[self.target_col_name],
                                      predicted_data=train_data_with_predictions[self.prediction_col_name])

        # evaluate model performance on test data
        test_MAPE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAPE,
                                       actual_data=test_data_with_predictions[self.target_col_name],
                                       predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_MSE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MSE,
                                      actual_data=test_data_with_predictions[self.target_col_name],
                                      predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_MAE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAE,
                                      actual_data=test_data_with_predictions[self.target_col_name],
                                      predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_r2 = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.R2,
                                     actual_data=test_data_with_predictions[self.target_col_name],
                                     predicted_data=test_data_with_predictions[self.prediction_col_name])

        training_result = {'params': 'base_model',
                           'train_MAPE': train_MAPE,
                           'test_MAPE': test_MAPE,
                           'train_MSE': train_MSE,
                           'test_MSE': test_MSE,
                           'train_MAE': train_MAE,
                           'test_MAE': test_MAE,
                           'train_r2': train_r2,
                           'test_r2': test_r2,
                           'fitting_duration': 0,
                           'series_type': 'base_model',
                           'trend_type': 'base_model',
                           'seasonality_components_no': 0,
                           'lags_no': 0

                           }

        print(f"train MAPE: {train_MAPE}")
        print(f"test MAPE: {test_MAPE}")
        print(f"train MSE: {train_MSE}")
        print(f"test MSE: {test_MSE}")
        print(f"train MAE: {train_MAE}")
        print(f"test MAE: {test_MAE}")
        print(f"train R2: {train_r2}")
        print(f"test R2: {test_r2}")

        # save train and test results
        train_data_with_predictions.to_csv(f'{self.results_path}/train_{self.dataset_name}')
        test_data_with_predictions.to_csv(f'{self.results_path}/test_{self.dataset_name}')

        return training_result
