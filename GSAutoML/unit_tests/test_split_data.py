import unittest
import pandas as pd
from GSAutoML.split_data.split_data import TimeSeriesSplitter  # Replace with the actual module name

class TestTimeSeriesSplitter(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
                                  'value': range(1, 101)})
        self.splitter = TimeSeriesSplitter(value_column="value")

    def test_train_test_split_no_validation(self):
        train_data, test_data = self.splitter.split_data(self.data)
        total_size = len(train_data) + len(test_data)
        self.assertEqual(total_size, len(self.data))
        self.assertEqual(len(train_data), int(len(self.data) * (1 - self.splitter.test_size)))

    def test_train_test_split_with_validation(self):
        validation_size = 0.1
        train_data, validation_data, test_data = self.splitter.split_data(self.data, validation_size=validation_size)
        total_size = len(train_data) + len(validation_data) + len(test_data)
        self.assertEqual(total_size, len(self.data))
        self.assertEqual(len(train_data), int(len(self.data) * (1 - self.splitter.test_size - validation_size)))
        self.assertEqual(len(validation_data), int(len(self.data) * validation_size))
