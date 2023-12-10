import pandas as pd
from scipy.stats import boxcox
from scipy.stats import boxcox_llf
from scipy.stats import chi2
from GizaAutoML.enums.time_series_types_enum import TimeSeriesTypesEnum

class SeriesTypeEstimator:
    """
    SeriesTypeEstimator class used for determining additive or multiplicative behavior of 
    time series data columns.
    This class provides a method to estimate whether the time series data in each column of
    a given DataFrame exhibits additive or multiplicative behavior based on 
    the log likelihood ratio test.
    Parameters:
        None

    Attributes:
        None

    Methods:
        series_type_estimator(data: pd.DataFrame) -> dict:
            Perform the log likelihood ratio test (comparing fitted BoxCox models with 
            the optimal lambda againist lamda of 1(no transformation)) 
            on each column of the DataFrame (excluding 'timestamp' and 'generator') to estimate 
            whether the data exhibits additive or multiplicative behavior.
            
            References: 
            https://grodri.github.io/glms/stata/c2s10
            https://en.wikipedia.org/wiki/Likelihood-ratio_test

    """
    def __init__(self):
        self.timestamp_col_name = "Timestamp"

    def series_type_estimator(self, dataframe: pd.DataFrame) -> dict:
        """
        Perform the log likelihood ratio test on each column of the DataFrame (excluding 'timestamp' and 'generator')
        to estimate whether the data exhibits additive or multiplicative behavior.

        Parameters:
            dataframe (pd.DataFrame): A pandas DataFrame containing the time series data with 'ds' column for timestamps
                                 and 'y' column for target values.

        Returns:
            dict: A dictionary where the keys are column names and the values are strings ('Additive' or 'Multiplicative')
                  indicating the estimated behavior of the corresponding column data.
        """
        data = dataframe.copy()
        results = {}  # Dictionary to store results
        timestamp_col = self.timestamp_col_name

        # Get column names (excluding 'timestamp' and 'generator')
        #columns_to_test = [col for col in data.columns if col not in [timestamp_col]]
        columns_to_test = data.select_dtypes(include=['int64', 'float64']).columns

        #looping through value columns to determine their type
        for column in columns_to_test:
            ts = data[column].dropna().values
            
            # Check if the lowest value in 'ts' is negative
            if ts.min() < 0:
                magnitude = abs(ts.min()) + 1
                ts = ts + magnitude
            else:
                ts = ts + 1
            
            transformed_data, lambda_param = boxcox(ts)
            
            llf_full = boxcox_llf(lambda_param, ts)
            llf_reduced = boxcox_llf(1, ts)
            
            test_statistic = -2 * (llf_reduced - llf_full)
            degrees_of_freedom = 1
            p_value = 1 - chi2.cdf(test_statistic, degrees_of_freedom)
            
            alpha = 0.05
            if p_value < alpha:
                result = TimeSeriesTypesEnum.MULTIPLICATIVE.value
            else:
                result = TimeSeriesTypesEnum.ADDITIVE.value
            
            results[column] = result
        
        return results
