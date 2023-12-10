import pandas as pd
from GizaAutoML.meta_feature_extraction.meta_feature_extractor_interface import MetaFeaturesExtractorInterface


class MetaFeaturesExtractorUnivariate(MetaFeaturesExtractorInterface):
    """ A class that has meta features extracting functions. """

    def __init__(self, raw_df: pd.DataFrame, imputed_df: pd.DataFrame, target_col: str = 'Target', timestamp_col: str = 'Timestamp'):
        super().__init__(raw_df, imputed_df, target_col, timestamp_col)

    def extract_meta_features(self) -> dict:
        meta_features = {'instances_no': self.get_num_instances(),
                         'missing_count_in_target': self.get_missing_count_in_target(),
                         'target_skewness': self.get_target_skewness(),
                         'target_kurtosis': self.get_target_kurtosis()
                         }
        return meta_features
