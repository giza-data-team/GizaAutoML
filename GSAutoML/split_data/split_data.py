import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

class TimeSeriesSplitter:
    def __init__(self, value_column: str,timestamp_column: str, test_size: float = 0.2):
        """
        Initialize the TimeSeriesSplitter.

        Parameters:
        - value_column (str): The name of the column containing the values.
        - test_size (float): The proportion of data to include in the test split.
        """
        self.value_column = value_column
        self.timestamp_column = timestamp_column
        self.test_size = test_size

    def split_data(self, data: pd.DataFrame, validation_size: float = 0.0):
        """
        Perform a train-test(optionally validation) split on time series data without shuffling.

        Parameters:
        - data (pd.DataFrame): The input time series data with a datetime column and value_column.
        - validation_size (float, optional): The proportion of data to include in the validation split.
                                             Default is 0.0 (no validation set).

        Returns:
        - tuple: Train, validation (if applicable), and test dataframes.
        """
        #sort the data by date
        data.sort_values(by=[self.timestamp_column], inplace=True)

        #slpitting the data with shuffle off to turn off random sampling
        train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=False)

        #if validation_size is provided then split to trains-validation-test
        if validation_size > 0.0:
            train_size = int(len(data) * (1 - self.test_size - validation_size))
            validation_size = int(len(data) * validation_size)
            train_data = data[:train_size]
            validation_data = data[train_size:train_size+validation_size]
            test_data = data[train_size+validation_size:]
            return train_data, validation_data, test_data
        else:
            test_data.reset_index(drop=True, inplace=True)
            return train_data, test_data

    def split_k_folds(self, data: pd.DataFrame):
        """
        Using TimeSeriesSplit from sklearn it performs k-fold splitting for time series data

        Parameters:
        - data (pd.DataFrame): The input time series data with a datetime column and value_column.

        Yields:
        - pd.DataFrame: Train and test dataframes.
        """
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_index, test_index in tscv.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            train_size = int(len(train_data) * (1 - self.test_size))
            train_data, test_data = train_data[:train_size], train_data[train_size:]

            yield train_data, test_data
