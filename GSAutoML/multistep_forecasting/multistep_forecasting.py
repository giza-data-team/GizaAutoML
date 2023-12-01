import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from GSAutoML.controller.common.utils import Utils
from datetime import datetime, timedelta


class MultistepTimeSeriesForecaster:
    def __init__(self, algorithm, num_steps, sampling_freq_min, num_lags,
                 validation_set=None,preprocessing_pipeline= None, target_col_name ="Target", timestamp_col_name="Timestamp"):
        """
        Initialize a MultistepTimeSeriesForecaster.

        Parameters:
        - algorithm: The regression algorithm to use for predictions.
        - num_steps: The number of steps to forecast.
        - validation_set: An optional validation set for evaluation.
        """
        self.algorithm = algorithm
        self.num_steps = num_steps
        self.sampling_freq_min = sampling_freq_min
        self.num_lags = num_lags
        self.validation_set = validation_set
        self.predictions = []
        self.preprocessing_pipeline = preprocessing_pipeline
        self.target_col_name = target_col_name
        self.timestamp_col_name = timestamp_col_name

    def preprocess_and_extract_features(self, data):
        """
        Preprocess and extract features from the input data.

        Parameters:
        - data: The input DataFrame containing 'Timestamp' and 'Target' columns.

        Returns:
        - data: The preprocessed DataFrame without the 'Timestamp' column.
        - pipeline: The preprocessing pipeline used for transformation.
        """
        data.columns = [self.timestamp_col_name, self.target_col_name]
        data[self.timestamp_col_name] = pd.to_datetime(data[self.timestamp_col_name])
        utils_instance = Utils()
        data, pipeline = utils_instance.get_processed_data(data)
        data = data.drop(self.timestamp_col_name, axis=1)
        return data, pipeline

    def predict(self, df):
        """
        Generate multistep predictions.

        Parameters:
        - df: The input DataFrame containing 'Timestamp' and 'Target' columns.

        Returns:
        - predictions: A list of multistep predictions.
        """
        self.train_data = df
        length = len(self.train_data)
        # Get the preprocessing pipeline to be used in each step
        if not self.preprocessing_pipeline:
            processed_data, pipeline = self.preprocess_and_extract_features(self.train_data)
            # get number of lags
            lagged_stage_index = -1
            lagged_stage = pipeline[1].stages[lagged_stage_index]
            num_lags = lagged_stage.col_lags_dic[self.target_col_name]['last_significant_lag_index']

        else:
            pipeline = self.preprocessing_pipeline
            num_lags = self.num_lags

        for step in range(self.num_steps):
            print(f"============== step {step} =================================")
            # Preprocess and extract features from the training data
            self.train_data = self.train_data.tail(num_lags+2)
            self.train_data.reset_index(drop=True, inplace=True)
            processed_data = pipeline.transform(self.train_data)
            print(processed_data)
            processed_data = processed_data.drop([self.timestamp_col_name,self.target_col_name], axis=1)
            # Create the next datetime value
            next_datetime = self.train_data[self.timestamp_col_name].iloc[-1] + timedelta(minutes=self.sampling_freq_min)

            # Make a prediction for the next step using the last row after preprocessing
            last_row_processed = processed_data.tail(1)
            next_step_prediction = self.algorithm.predict(last_row_processed)
            self.predictions.append(next_step_prediction[0])

            # Append the prediction to the training data for the next step
            self.train_data = pd.concat(
                [self.train_data, pd.DataFrame({self.timestamp_col_name: [next_datetime], self.target_col_name: [next_step_prediction[0]]})],ignore_index=True)
            # if step == 0:
            #     processed_data
        return self.predictions

    def validate_prediction(self, prediction, df):
        # Get the length of the prediction array
        n = len(prediction)

        # Check if the 'Target' column has at least 'n' values
        if len(df[self.target_col_name]) < n:
            raise ValueError(f"{self.target_col_name} column has less than {n} values.")
        df = self.preprocessing_pipeline[0].stages[0].transform(df)
        # Extract the true values from the 'Target' column
        true_values = df[self.target_col_name].values[:n]
        prediction = np.array(prediction)
        # Calculate metrics
        mae = mean_absolute_error(true_values, prediction)
        mse = mean_squared_error(true_values, prediction)
        mape = np.mean(np.abs((true_values - prediction) / true_values)) * 100
        r2 = r2_score(true_values, prediction)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "MAPE": mape,
            "R2": r2
        }
        return metrics




# # Example usage:
# if __name__ == "__main__":
#     # Read the CSV file using pandas
#     df = pd.read_csv('600.csv')
#     df_test = pd.read_csv('test_Adaboost_600.csv')
#     num_steps = 3
#     # train a lasso regressor to act as the best algorithm
#     lasso_regressor = Lasso(alpha=0.1)
#     forecaster = MultistepTimeSeriesForecaster(lasso_regressor, num_steps=num_steps)
#     processed_data, pipeline = preprocess_and_extract_features(df)
#     lasso_regressor = lasso_regressor.fit(processed_data.drop(columns=['Target'], axis=1), processed_data['Target'])
#     # initialize the multistep class
#     forecaster = MultistepTimeSeriesForecaster(lasso_regressor, num_steps=num_steps)
#     # Make predictions
#     predictions = forecaster.predict(df)
#     # validattion
#     print(validate_prediction(predictions, df_test))
