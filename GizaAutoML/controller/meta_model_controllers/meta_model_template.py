import time
from abc import abstractmethod, ABC
import pandas as pd

from GizaAutoML.controller.common.knowledge_base_collector import KnowledgeBaseCollector
from GizaAutoML.controller.meta_features_controller import MetaFeaturesController
from GizaAutoML.controller.common.utils import Utils
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GizaAutoML.meta_model.meta_model_algorithms_recommender import MetaModelAlgorithmsRecommender
from GizaAutoML.meta_model.optimization.Generate_initial_search_space import ConfigSpaceBuilder
from GizaAutoML.controller.meta_model_controllers.meta_model_utils import MetaModelUtils


class MetaModelTemplate(ABC):
    """
        Template Design Pattern for training and evaluating meta-models for regression on time series data.

        This class provides Basic template functionality for training meta-models on a given time series dataset,
        optimizing hyperparameters, and selecting the best-performing algorithm.

        Args:
            dataset_instance (Dataset): The dataset instance containing information about the dataset.
            dataframe (DataFrame): The dataset as a DataFrame.
            sorting_metric (str): The metric used for sorting and ranking algorithms.
            time_budget (int): The time budget (in minutes) for training the meta-model.
            save_results (bool): Flag indicating whether to save the results.
            random_seed (int): Random seed for reproducibility.

        """

    def __init__(self, raw_dataframe,
                 processed_dataframe=pd.DataFrame(),
                 sorting_metric=RegressionScoringMetricEnum.MAE.name,
                 time_budget=10,
                 save_results=True,
                 random_seed=1,
                 target_col="Target",
                 timestamp_col="Timestamp",
                 dataset_name=None,
                 dataset_instance=None,
                 utils=None):
        """
        Initialize the MetaModel Template.
        :param (DataFrame) raw_dataframe:  The raw dataset as a DataFrame.
        :param (DataFrame) processed_dataframe: The preprocessed dataset as a DataFrame if available.
        :param (str) sorting_metric:  The metric used for sorting and ranking algorithms.
        :param (int) time_budget: The time budget (in minutes) for training the meta-model.
        :param (bool) save_results: Flag indicating whether to save the results.
        :param (int) random_seed: Random seed for reproducibility.
        :param (str) target_col: The name of the target column.
        :param (str) timestamp_col: The name of the date column.
        :param (str) dataset_name: The name of the target column.
        :param (dict) dataset_instance: The name of the target column.
        """
        self.dataframe = raw_dataframe
        self.processed_dataframe = processed_dataframe
        self.sorting_metric = sorting_metric
        self.time_budget = time_budget
        self.target_col = target_col
        self.timestamp_col_name = timestamp_col
        if not utils:
            self.utils = Utils(series_col=self.target_col)
        else:
            self.utils = utils
        self.exceptions = []
        self.save_results_flag = save_results
        self.random_seed = random_seed
        self.dataset_instance = dataset_instance
        self.kb_collector = KnowledgeBaseCollector()
        self.dataset_name = dataset_name
        self.meta_model_utils = MetaModelUtils(time_stamp_col_name=self.timestamp_col_name,
                                               target_col_name=self.target_col,
                                               random_seed=self.random_seed)

    def get_resampled_df(self, dataframe):
        """
                Returns a resampled dataframe.
                :param (DataFrame) dataframe:  The  DataFrame to be resampled.
                :return: resampled dataframe
        """
        resampled_df = self.utils.resample_data(dataframe)
        return resampled_df

    def get_series_meta_features(self, dataframe, dataset_instance):
        """
                Returns meta features of dataframe.
                :param (DataFrame) dataframe:  The  DataFrame to get the meta features for.
                :param (DataFrame) dataset_instance:  The  dataset instance of the dataframe.
                :return: resampled dataframe
        """
        series_meta_features = MetaFeaturesController(dataframe, dataset_instance,
                                                      save_results=self.save_results_flag,
                                                      target_col=self.target_col).meta_features
        return series_meta_features

    @abstractmethod
    def get_split_data(self, dataframe):
        """
        abstract method for splitting data based on saving flag
        """
        pass

    def get_recommended_algorithms_info(self, dataframe, series_meta_features):
        """
                Returns meta features of dataframe.
                :param (DataFrame) dataframe:  The  DataFrame to get recommended algorithm for.
                :param (dict) series_meta_features:  meta features dict of the dataframe.
                :return: algorithm recommender object & algortihms info dict
        """

        algorithms_recommender = MetaModelAlgorithmsRecommender(series=dataframe,
                                                                series_meta_features=series_meta_features,
                                                                sorting_metric=self.sorting_metric,
                                                                target_col_name=self.target_col,
                                                                time_stamp_col_name=self.timestamp_col_name)
        top_3_algorithms = algorithms_recommender.get_recommended_algorithms()
        top_3_algorithms_configs = algorithms_recommender.get_algorithms_configuration(top_3_algorithms)

        algorithms_info = self.meta_model_utils.get_algorithm_search_space(top_3_algorithms_configs)
        return algorithms_recommender, algorithms_info

    def get_series_type(self, dataframe):
        """
                Returns the series types of features in dataframe.
                :param (DataFrame) dataframe:  The  DataFrame to get the series type for.
                :return: series types dict
        """

        series_type = self.utils.get_series_type(dataframe)
        return series_type

    def get_processed_data(self, dataframe, series_type):
        """
                perform preprocessing pipeline.
                :param (DataFrame) dataframe:  The  DataFrame fit the pipeline on.
                :param (dict) series_type: series types of features in dataframe
                :return: processed data and the fitted pipeline
        """
        dataframe = dataframe if self.processed_dataframe.empty else self.processed_dataframe
        processed_train_data, fitted_processing_pipeline = self.utils. \
            get_processed_data(dataframe, series_type=series_type, preprocessing=self.processed_dataframe.empty)
        processed_train_data.set_index('Timestamp', inplace=True)
        return processed_train_data, fitted_processing_pipeline

    def get_meta_models_info(self, processed_train_data, algorithms_info):
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
        search_spaces = [v['search_space'] for v in algorithms_info.values()]
        defaults = [v['defaults'] for v in algorithms_info.values()]
        algorithms_list = list(algorithms_info.keys())
        configurations_list = config_space_builder.build_config_space(search_spaces, defaults)

        # optimize the model with the default configurations and the configuration space
        meta_models_info = self.meta_model_utils.hyper_opt(x, y, algorithms_list,
                                                           configurations=configurations_list,
                                                           default_configurations=defaults,
                                                           time_budget=self.time_budget,
                                                           random_seed=self.random_seed)
        return meta_models_info, algorithms_list

    def get_best_algorithm(self, meta_models_info, algorithms_list):
        """
                return the best algorithm with its hyperparameter
                :param (dict) meta_models_info: dict containing information about the hyperparameter optimization
                                     results for each algorithm.
                :param (list) algorithms_list: list of algorithms names
                :return: meta_model info dict and algorithms list
        """
        meta_models_info_costs = [v['min_cost'] for v in meta_models_info.values()]
        print(f"min costs: {meta_models_info_costs}")
        print(f"best algorithm cost: {min(meta_models_info_costs)}")
        min_index = meta_models_info_costs.index(min(meta_models_info_costs))
        best_algorithm_name = algorithms_list[min_index]
        best_hyperparameters = meta_models_info[best_algorithm_name]['hyperparameters']
        trials_no = meta_models_info[best_algorithm_name]['trials_no']
        print(f"best selected algorithm: {best_algorithm_name}")
        print(f"best hyperparameters: {best_hyperparameters}")
        return best_algorithm_name, best_hyperparameters, trials_no

    def get_fitted_model(self, processed_dataframe, series_type, best_algorithm_name,
                         best_hyperparameters, algorithms_recommender: MetaModelAlgorithmsRecommender):
        """
            fit the chosen algorithm with the best hyperparameters to processed dataframe
            :param (DataFrame) processed_dataframe: processed data
            :param (dict) series_type: series types of features in dataframe
            :param (str) best_algorithm_name: algorithm name to train the model with
            :param (dict) best_hyperparameters: best hyperparameters to train the model with
            :param (object) algorithms_recommender: algorithms recommender object
            :return: fitted model piplne and fitting duration  and dict contain training results
        """
        start_time = time.time()
        training_results, fitted_modeling_pipeline, train_data_with_predictions = \
            algorithms_recommender.evaluate_algorithm(df=processed_dataframe,
                                                      series_type=series_type,
                                                      algorithm_name=best_algorithm_name,
                                                      hyperparameters=best_hyperparameters,
                                                      target_col=self.target_col)
        end_time = time.time()
        fitting_duration = end_time - start_time
        return training_results, fitted_modeling_pipeline, fitting_duration

    def get_extracted_features_info(self, fitted_processing_pipeline):
        """
                    extract meta info from the time feature extraction stages
                    :param (pipline) fitted_processing_pipeline: preprocessing pipline with feature extraction stages
                    :return: dict with extracted_features_info
                """
        extracted_features_info = {}
        lagged_stage_index = self.utils.features_indexes['lagged_stage_index']
        trend_stage_index = self.utils.features_indexes['trend_stage_index']
        seasonality_stage_index = self.utils.features_indexes['seasonality_stage_index']
        # concat lags in test data
        if lagged_stage_index:
            lagged_stage = fitted_processing_pipeline[self.utils.features_extraction_stage_index].stages[lagged_stage_index]
            last_significant_lags_index = lagged_stage.col_lags_dic[self.target_col]['last_significant_lag_index']
            significant_lags_no = lagged_stage.col_lags_dic[self.target_col]['significant_lags']
            extracted_features_info['lags_no'] = significant_lags_no
            extracted_features_info['last_significant_lags_index'] = last_significant_lags_index
        if trend_stage_index:
            trend_stage = fitted_processing_pipeline[self.utils.features_extraction_stage_index].stages[trend_stage_index]
            trend_type = trend_stage.trend_type
            extracted_features_info['trend_type'] = trend_type
        if seasonality_stage_index:
            # get no of seasonality components
            seasonality_stage = fitted_processing_pipeline[self.utils.features_extraction_stage_index].stages[seasonality_stage_index]
            seasonality_components_no = len(seasonality_stage.get_peak_frequencies())
            extracted_features_info['seasonality_components_no'] = seasonality_components_no
        return extracted_features_info

    def build_parent_pipline(self, pipelines: list):
        parent_pipeline = self.utils.create_pipeline(pipelines)
        return parent_pipeline

    @abstractmethod
    def get_test_results(self, train_data, test_data, last_significant_lags_index, parent_pipeline,
                         algorithms_recommender):
        pass

    @abstractmethod
    def get_training_results(self, best_hyperparameters, training_results, test_result, fitting_duration,
                             extracted_features_info, best_algorithm_name, series_type, training_time):
        pass

    def run(self):
        start_time = time.time()

        if self.utils.check_if_univariate(self.dataframe):
            df = self.utils.prepare_univariate_data(self.dataframe)
        else:
            df = self.dataframe
        if not self.processed_dataframe.empty:
            # apply resampling if preprocessing will be through the engine
            series_df = self.get_resampled_df(df)
        else:
            series_df = df.copy()
        series_type = self.get_series_type(series_df)
        series_meta_features = self.get_series_meta_features(series_df, self.dataset_instance)
        train_data, test_data = self.get_split_data(series_df)
        algorithms_recommender, algorithms_info = self.get_recommended_algorithms_info(train_data, series_meta_features)
        processed_train_data, fitted_processing_pipeline = self.get_processed_data(series_df, series_type)
        meta_models_info, algorithms_list = self.get_meta_models_info(processed_train_data, algorithms_info)
        best_algorithm_name, best_hyperparameters, trials_no = self.get_best_algorithm(meta_models_info,
                                                                                       algorithms_list)
        training_results, fitted_modeling_pipeline, fitting_duration = self.get_fitted_model(processed_train_data,
                                                                                             series_type,
                                                                                             best_algorithm_name,
                                                                                             best_hyperparameters,
                                                                                             algorithms_recommender)
        self.parent_pipeline = self.build_parent_pipline([fitted_processing_pipeline, fitted_modeling_pipeline])
        self.extracted_features_info = self.get_extracted_features_info(fitted_processing_pipeline)
        no_lags = self.extracted_features_info['last_significant_lags_index'] if self.extracted_features_info else None
        test_results = self.get_test_results(train_data, test_data,
                                             no_lags,
                                             self.parent_pipeline,
                                             algorithms_recommender)
        self.training_results_dict = self.get_training_results(best_hyperparameters, training_results,
                                                               test_results, fitting_duration,
                                                               self.extracted_features_info,
                                                               best_algorithm_name, series_type,
                                                               training_time=time.time() - start_time)
