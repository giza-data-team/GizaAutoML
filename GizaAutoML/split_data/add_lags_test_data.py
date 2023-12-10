import pandas as pd

class LaggedDataPreprocessor:
    def __init__(self, train_data, test_data, num_lags):
        self.train_data = train_data
        self.test_data = test_data
        self.num_lags = num_lags

    def preprocess_data(self):
        train_last_rows = self.train_data.tail(self.num_lags)
        processed_test_data = pd.concat([train_last_rows, self.test_data], ignore_index=True)
        return processed_test_data
