from datetime import datetime
import numpy as np
import pandas as pd
import unittest

from GSAutoML.feature_engineering.data_preproccessing.estimators.prophet_imputer_estimator import ProphetImputerEstimator


class TestProphetImputer(unittest.TestCase):

    def setUp(self) -> None:
        data_dict = {'Timestamp': [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3),
                                   datetime(2021, 1, 4), datetime(2021, 1, 5)],
                     "sensor1": [20, np.NaN, 30, 23, 20],
                     "sensor2": [330.0, 344.0, 7600.0, np.NaN, 4000.0],
                     "sensor3": [1, 2, np.NaN, 4, 5],
                     "sensor4": [1, 23, 25, 43, 52]}
        predict_dict = {'Timestamp': [datetime(2021, 1, 6)],
                        "sensor1": [20],
                        "sensor2": [np.NaN],
                        "sensor3": [5],
                        "sensor4": [np.NaN]}
        self.predict = pd.DataFrame(predict_dict)
        self.df = pd.DataFrame(data_dict)
        self.timestamp_col = 'Timestamp'

    def test_prophet_imputer(self):
        prophet_imputer_input_cols = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
        prophet_imputer_es = ProphetImputerEstimator(
            timestamp_col=self.timestamp_col,
            input_cols=prophet_imputer_input_cols,
            seasonality_mode={"sensor1": "additive", "sensor2": "additive", "sensor3": "multiplicative",
                              "sensor4": "additive"}
        )
        model = prophet_imputer_es.fit(self.df)
        self.assertIsInstance(model, ProphetImputerEstimator)
        df_after = model.transform(self.df)
        cols = df_after.columns
        self.assertIn('sensor1', cols)
        self.assertIn('sensor2', cols)
        self.assertIn('sensor3', cols)
        self.assertIn('sensor4', cols)
        self.assertFalse(df_after['sensor1'].hasnans)
        self.assertFalse(df_after['sensor2'].hasnans)
        self.assertFalse(df_after['sensor3'].hasnans)
        self.assertFalse(df_after['sensor4'].hasnans)
        # test that all not null values Maintain same order after imputation
        for input_col in prophet_imputer_input_cols:
            non_null_index = list(self.df[self.df[input_col].notna()].index)
            self.assertTrue(self.df.iloc[non_null_index][input_col].equals(df_after.iloc[non_null_index][input_col]))

    def test_prophet_predict(self):
        prophet_imputer_input_cols = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
        prophet_imputer_es = ProphetImputerEstimator(
            timestamp_col=self.timestamp_col,
            input_cols=prophet_imputer_input_cols,
            seasonality_mode={"sensor1": "additive", "sensor2": "additive", "sensor3": "multiplicative",
                              "sensor4": "additive"}
        )
        model = prophet_imputer_es.fit(self.df)
        self.assertIsInstance(model, ProphetImputerEstimator)
        print(self.predict)
        df_after = model.transform(self.predict)
        print(df_after)
        cols = df_after.columns
        self.assertIn('sensor1', cols)
        self.assertIn('sensor2', cols)
        self.assertIn('sensor3', cols)
        self.assertIn('sensor4', cols)
        self.assertFalse(df_after['sensor1'].hasnans)
        self.assertFalse(df_after['sensor2'].hasnans)
        self.assertFalse(df_after['sensor3'].hasnans)
        self.assertFalse(df_after['sensor4'].hasnans)
