import pandas as pd

from GizaAutoML.controller.meta_model_controllers.meta_model_template import MetaModelTemplate
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum


class MetaModelWithOutSaving(MetaModelTemplate):

    def __int__(self, raw_dataframe, processed_dataframe=pd.DataFrame(),
                sorting_metric=RegressionScoringMetricEnum.MAE.name,
                save_results=False,
                time_budget=10,
                random_seed=1,
                target_col="Target",
                dataset_name=None,
                dataset_instance=None,
                utils=None):
        super().__init__(raw_dataframe, processed_dataframe, sorting_metric, time_budget, save_results, random_seed, target_col,
                         dataset_name,
                         dataset_instance,
                         utils)

    def get_split_data(self, resampled_df):
        train_data, test_data = resampled_df, None
        return train_data, test_data

    def get_training_results(self, best_hyperparameters, training_results, test_result, fitting_duration,
                             extracted_features_info, best_algorithm_name, series_type,
                             training_time):
        print("Training Results Are Only Available With Saving Mode On")

    def get_test_results(self, train_data, test_data, last_significant_lags_index, parent_pipeline,
                         algorithms_recommender):
        print("Test Results Are Only Available With Saving Mode On")
