import numpy as np
from statsmodels.tsa.stattools import adfuller


class FeatureExtractionUtils:
    @staticmethod
    def check_stationary(feature, significant_value):
        """
        perform Augmented Dickey-Fuller test to check the stationary status of the feature
        :param feature: feature in the data that we want to check
        :return Boolean flag to indicate if the feature is stationary or non-stationary
        """
        # use to_numpy() or .values as the AdFuller requires  the input to be a 1D array
        feature = feature.to_numpy()
        feature = feature[~np.isnan(feature)]
        result = adfuller(feature)

        stationary = result[1] <= significant_value
        return stationary, result

    def apply_differencing(self, dataframe, non_stationary_features, stationary_col_names,
                           period: int = 1,
                           write_to_df=False):
        """
        Apply the differencing on the non-stationary features
        :param dataframe Pandas Dataframe
        :param non_stationary_features: A list contains the names of the non-stationary features in the data.
        :param stationary_col_names: A list of the names of the output columns.
        :param period Integer of the period by which we will apply the differencing.
        :param write_to_df Boolean to write the new features to the dataframe or not.
        :return Tuple(Pandas Series of the extracted differencing features,
        List containing the names of the columns that are still non-stationary after applying the differencing.)
        """
        for col in zip(non_stationary_features, stationary_col_names):

            new_feature = dataframe[col[0]].diff(periods=period)
            if write_to_df:
                dataframe[col[1]] = new_feature
        return dataframe
