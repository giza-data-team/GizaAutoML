import pandas as pd
from GSAutoML.feature_engineering.feature_extraction.feature_extraction_utils import \
    FeatureExtractionUtils
from GSAutoML.feature_engineering.common.transformer_interface import ITransformer


class StationaryFeaturesTransformer(ITransformer):
    def __init__(self, significant_value: float = None, diff_features: dict = None,
                 adf_test_results=None):
        super().__init__()
        self.set_significant_value(significant_value)
        self.set_diff_features(diff_features)
        self.set_adf_test_results(adf_test_results)

    def set_significant_value(self, significant_value):
        self.significant_value = significant_value

    def get_significant_value(self):
        return self.significant_value

    def set_diff_features(self, diff_features):
        self.diff_features = diff_features

    def get_diff_features(self):
        return self.diff_features

    def set_adf_test_results(self, adf_test_results):
        self.adf_test_results = adf_test_results

    def get_adf_test_results(self):
        return self.adf_test_results

    def fit(self, X, y=None):
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print(">>>>>>>>>> In stationary transformer >>>>>>>>>>>>>>>>>")

        diff = {}  # store columns with differencing order of each column.
        self.max_order = 0
        self.initial_numbers = {}
        diff_features = self.get_diff_features()
        # Sample Pandas Series with NaN values

        for col in diff_features.keys():
            try:
                diff[diff_features[col]].append(col)
            except:
                diff[diff_features[col]] = [col]
            self.max_order = max(self.max_order, diff_features[col])
        for col in diff_features:
            for differencing_order in range(1, diff_features[col] + 1):

                # Find the index of the first non-NaN value
                first_non_nan_index = dataframe.sort_index()[col].first_valid_index()
                first_non_nan_value = dataframe.loc[first_non_nan_index, col]
                if col in self.initial_numbers:
                    self.initial_numbers[col].update({differencing_order: first_non_nan_value})
                else:
                    self.initial_numbers[col] = {differencing_order: first_non_nan_value}

                dataframe = FeatureExtractionUtils().apply_differencing(dataframe, non_stationary_features=[col],
                                                                        stationary_col_names=[col],
                                                                        write_to_df=True)

        dataframe = dataframe.dropna()
        return dataframe
