import pandas as pd
from GSAutoML.feature_engineering.common.estimator_interface import IEstimator
from GSAutoML.feature_engineering.feature_extraction.feature_extraction_utils import \
    FeatureExtractionUtils
from GSAutoML.feature_engineering.feature_extraction.transformers.stationary_features_transformer import \
    StationaryFeaturesTransformer


class StationaryFeaturesEstimator(IEstimator):

    def __init__(self, cols=None):
        super().__init__()
        if cols is None:  # If no columns are specified applied the stationarity on the target column only
            self.cols = ["Target"]
        else:
            self.cols = cols

        self.adf_test_results = None
        self.sampling_threshold = 90
        self.stationary_max_diff = 2
        self.significant_value = 0.05

    def get_diff_features(self):
        return self._schema_extracted_features

    def get_adf_test_results(self):
        return self.adf_test_results

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_stationary_features_instance()

    def fit(self, dataframe: pd.DataFrame, y=None):
        """
        Perform the AdFuller test and apply differencing to the non-stationary features.
        """
        print(">>>>>>>>>> In stationary estimator >>>>>>>>>>>>>>>>>")

        dataframe = dataframe.copy()
        self._schema_extracted_features = {}
        adf_test_results = {}
        # check the length of the samples
        if self.sampling_threshold > dataframe.shape[0]:
            sample_threshold = dataframe.shape[0]
        else:
            sample_threshold = self.sampling_threshold
        # check if the features are stationary
        max_differencing_order = self.stationary_max_diff
        for col in self.cols:
            if dataframe.dtypes[col] in ["object", ]:
                continue
            stationary_flag = False
            differencing_order = 1
            while differencing_order <= max_differencing_order and not stationary_flag:

                start_index = sample_threshold if sample_threshold < dataframe.shape[0] else dataframe.shape[0]
                stationary_flag, adf_result = FeatureExtractionUtils().check_stationary(
                    dataframe[col].iloc[-start_index:],
                    self.significant_value)

                # add result of each col to the results artifact
                adf_test_results[col] = adf_result
                if not stationary_flag:
                    col_differencing_order = self._schema_extracted_features.get(col, 0)
                    self._schema_extracted_features[col] = col_differencing_order + 1
                    dataframe = FeatureExtractionUtils().apply_differencing(dataframe,
                                                                            [col], [col],
                                                                            write_to_df=True)
                    differencing_order += 1
                else:
                    break

        self.max_order = max(self._schema_extracted_features.values(), default=0)
        self.adf_test_results = adf_test_results
        return self

    def _get_initial_numbers(self):
        return self.stationary_transformer.initial_numbers

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.stationary_transformer = StationaryFeaturesTransformer(
            self.significant_value,
            self._schema_extracted_features,
            self.adf_test_results)
        return self.stationary_transformer.transform(X)

    def postprocess(self, dataframe_with_predictions,col_name):
        for key, value in self._get_initial_numbers().items():
            if key == col_name:
                for order in value:
                    dataframe_with_predictions["prediction"] = dataframe_with_predictions.sort_index()[
                                                                   "prediction"].cumsum() + value[order]
            else:
                continue
        return dataframe_with_predictions
