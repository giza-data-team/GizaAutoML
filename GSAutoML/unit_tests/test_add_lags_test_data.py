import unittest
import pandas as pd
from GSAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor  # Assuming you have the class in a file named lagged_data_preprocessor.py

class TestLaggedDataPreprocessor(unittest.TestCase):
    def test_preprocess_data(self):
        train_data = pd.DataFrame({'timestamp': ['2023-08-10', '2023-08-11', '2023-08-12', '2023-08-13', '2023-08-14'],
                                   'value': [10, 20, 30, 40, 50]})
        test_data = pd.DataFrame({'timestamp': ['2023-08-15', '2023-08-16', '2023-08-17'],
                                  'value': [60, 70, 80]})
        num_lags = 3

        preprocessor = LaggedDataPreprocessor(train_data, test_data, num_lags)
        processed_test_data = preprocessor.preprocess_data()

        expected_data = pd.DataFrame({'timestamp': ['2023-08-12', '2023-08-13', '2023-08-14', '2023-08-15', '2023-08-16', '2023-08-17'],
                                      'value': [30, 40, 50, 60, 70, 80]})

        pd.testing.assert_frame_equal(processed_test_data, expected_data)
