import pandas as pd

from GSAutoML.controller.meta_model_controllers.meta_model_template import MetaModelTemplate
from GSAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GSAutoML.meta_model.meta_model_algorithms_recommender import MetaModelAlgorithmsRecommender
from GSAutoML.meta_model.optimization.Generate_initial_search_space import ConfigSpaceBuilder
from GSAutoML.controller.meta_model_controllers.meta_model_utils import MetaModelUtils


class CurrentRegression(MetaModelTemplate):
    """
    The Concrete Classes for MetaModelTemplate that overrides steps used in Current State Regression model
    """
    def __int__(self, raw_dataframe, processed_dataframe=pd.DataFrame(),
                sorting_metric=RegressionScoringMetricEnum.MAE.name,
                save_results=False,
                time_budget=10,
                random_seed=1,
                target_col="Target",
                timestamp_col="Timestamp",
                dataset_name=None,
                dataset_instance=None,
                utils=None):
        super().__init__(raw_dataframe, processed_dataframe, sorting_metric, time_budget, save_results, random_seed,
                         target_col,
                         dataset_name,
                         dataset_instance, utils)
        self.meta_model_utils = MetaModelUtils(time_stamp_col_name=self.timestamp_col_name,
                                               target_col_name=self.target_col,
                                               random_seed=self.random_seed)

    def get_split_data(self, dataframe):
        train_data, test_data = dataframe, None
        return train_data, test_data

    def get_series_meta_features(self, dataframe, dataset_instance):
        """ overwrite the series meta features as meta model isn't supported here """
        return None

    def get_resampled_df(self, dataframe):
        """ overwrite apply resampling method as resampling isn't supported in current predictions """

        return dataframe

    def get_recommended_algorithms_info(self, dataframe, series_meta_features=None):
        """
                Returns meta features of dataframe.
                :param (DataFrame) dataframe:  The  DataFrame to get recommended algorithm for.
                :param (dict) series_meta_features:  meta features dict of the dataframe.
                :return: algorithm recommender object & algortihms info dict
        """

        algorithms_recommender = MetaModelAlgorithmsRecommender(series=dataframe,
                                                                series_meta_features=series_meta_features,
                                                                sorting_metric=self.sorting_metric)
        return algorithms_recommender, None

    def get_meta_models_info(self, processed_train_data, algorithms_info=None):
        """
                perform hyperparameters optimization to get the meta_model info.
                :param (DataFrame) processed_train_data:  The  DataFrame used to get the best hyperparameters for
                :param (list) algorithms_info: list of algorithms names
                :return: meta_model info dict and algorithms list
        """
        y = processed_train_data[self.target_col]
        x = processed_train_data.drop(self.target_col, axis=1)
        config_space_builder = ConfigSpaceBuilder(random_seed=self.random_seed)
        # list of config spaces
        algorithms_list = ['RandomForestRegressor', 'LassoRegressor', 'AdaboostRegressor']
        configurations_list = config_space_builder.current_step_conf_space()

        # optimize the model with the default configurations and the configuration space
        meta_models_info = self.meta_model_utils.hyper_opt(x, y, algorithms_list,
                                                           configurations=configurations_list,
                                                           default_configurations=[None, None, None],  # TODO
                                                           time_budget=self.time_budget,
                                                           random_seed=self.random_seed)
        return meta_models_info, algorithms_list

    def get_extracted_features_info(self, fitted_processing_pipeline):

        return None

    def get_test_results(self, train_data, test_data, last_significant_lags_index, parent_pipeline,
                         algorithms_recommender):
        print("Test Results Are Not Available With Current State Regression")

    def get_training_results(self, best_hyperparameters, training_results, test_result, fitting_duration,
                             extracted_features_info, best_algorithm_name, series_type,
                             training_time):
        """ no meta model supported so no available knowledge base  """
        print("Training Results Are Not Available With Current State Regression")

