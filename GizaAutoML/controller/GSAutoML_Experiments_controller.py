import time
import pandas as pd
from GizaAutoML.controller.meta_features_controller import MetaFeaturesController
from GizaAutoML.controller.common.utils import Utils
from GizaAutoML.enums.algorithms_mapping import AlgorithmsMappingEnum
from GizaAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GizaAutoML.meta_model.meta_model_algorithms_recommender import MetaModelAlgorithmsRecommender
from GizaAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GizaAutoML.split_data.split_data import TimeSeriesSplitter
from GizaAutoML.meta_model.optimization.Generate_initial_search_space import ConfigSpaceBuilder
from GizaAutoML.controller.common.use_case_predictor_factory import PredictorFactory
from GizaAutoML.enums.ML_tasks_enum import MLTasksEnum
from GizaAutoML.data_modeling.evaluator import Evaluator
from GizaAutoML.controller.meta_model_controllers.meta_model_utils import MetaModelUtils


class GSAutoMLController:
    """
        Controller for training and evaluating meta-models for regression on time series data.

        This class provides functionality for training meta-models on a given time series dataset,
        optimizing hyperparameters, and selecting the best-performing algorithm.

        Args:
            dataset_instance (Dataset): The dataset instance containing information about the dataset.
            dataframe (DataFrame): The dataset as a DataFrame.
            sorting_metric (str): The metric used for sorting and ranking algorithms.
            time_budget (int): The time budget (in minutes) for training the meta-model.
            save_results (bool): Flag indicating whether to save the results.
            random_seed (int): Random seed for reproducibility.

        Attributes:
            dataset_instance (Dataset): The dataset instance containing information about the dataset.
            dataset_name (str): The name of the dataset.
            dataframe (DataFrame): The dataset as a DataFrame.
            sorting_metric (str): The metric used for sorting and ranking algorithms.
            time_budget (int): The time budget (in minutes) for training the meta-model.
            utils (Utils): An instance of the utility class for common operations.
            save_results_flag (bool): Flag indicating whether to save the results.
            random_seed (int): Random seed for reproducibility.

        Methods:
            get_algorithm_search_space(self, top_3_algorithms):
                Get the search space for the top-performing algorithms.

            hyper_opt(self, x, y, algorithms, configurations, default_configurations):
                Perform hyperparameter optimization for multiple algorithms.

            run(self):
                Run the GizaAutoML training and evaluation process.
        """

    def __init__(self, dataset_instance, dataframe, sorting_metric,
                 time_budget, save_results, random_seed, time_stamp_col_name,
                 target_col_name, results_path, multi_step_flag: bool, is_forecast=True):
        self.dataset_instance = dataset_instance  # instance from Datasets Table
        self.dataset_name = self.dataset_instance.name
        self.dataframe = dataframe
        self.sorting_metric = sorting_metric
        self.time_budget = time_budget
        self.is_forecast = is_forecast
        self.save_results_flag = save_results
        self.random_seed = random_seed
        self.time_stamp_col_name = time_stamp_col_name
        self.target_col_name = target_col_name
        self.results_path = results_path
        self.multi_step_flag = multi_step_flag
        self.prediction_col_name = 'prediction'
        self.evaluation_metric_enum = RegressionEvaluationMetricEnum
        self.utils = Utils(is_forecast=self.is_forecast)
        self.meta_model_utils = MetaModelUtils(time_stamp_col_name=self.time_stamp_col_name,
                                               target_col_name=self.target_col_name,
                                               random_seed=self.random_seed)

    def run(self):
        """
        Run the GizaAutoML training and evaluation process.

        This method performs the entire meta-model training and evaluation process, including data preprocessing,
        algorithm selection, hyperparameter optimization, and evaluation on training and test data.

        Returns:
            training_result (dict): A dictionary containing training results, including hyperparameters and
                                    performance metrics on the training data.

        """
        print(f"=============  Start Meta Model Training for {self.dataset_name} ========================")
        if self.utils.check_if_univariate(self.dataframe):
            df = self.utils.prepare_univariate_data(self.dataframe)
        else:
            df = self.dataframe
        # resample data
        resampled_df = self.utils.resample_data(df)
        # get meta_features of the new dataset
        series_meta_features = MetaFeaturesController(resampled_df, self.dataset_instance,
                                                      save_results=self.save_results_flag).meta_features

        # split data
        train_data, test_data = TimeSeriesSplitter(value_column=self.target_col_name,
                                                   timestamp_column=self.time_stamp_col_name).split_data(
            data=resampled_df)

        # get top algorithms on train data only
        algorithms_recommender = MetaModelAlgorithmsRecommender(series=train_data,
                                                                series_meta_features=series_meta_features,
                                                                sorting_metric=self.sorting_metric)

        # get top 3 algorithms with their hyperparameters configurations
        top_3_algorithms = algorithms_recommender.get_recommended_algorithms()
        top_3_algorithms_configs = algorithms_recommender.get_algorithms_configuration(top_3_algorithms)

        print(top_3_algorithms_configs)
        print("============= Optimization on train data using SMAC ======================")

        # get search space from grid search algorithm
        # use algorithm hyperparameters as defaults
        algorithms_info = self.meta_model_utils.get_algorithm_search_space(top_3_algorithms_configs)
        print(algorithms_info)

        # apply preprocessing and feature extraction on train data
        series_type = self.utils.get_series_type(resampled_df)
        # preprocess train data
        processed_train_data, fitted_processing_pipeline = self.utils.get_processed_data(train_data)
        processed_train_data.set_index(self.time_stamp_col_name, inplace=True)

        y = processed_train_data[self.target_col_name]
        x = processed_train_data.drop(self.target_col_name, axis=1)

        # build configuration space for the selected algorithm
        config_space_builder = ConfigSpaceBuilder(random_seed=self.random_seed)
        # list of config spaces
        search_spaces = [v['search_space'] for v in algorithms_info.values()]
        defaults = [v['defaults'] for v in algorithms_info.values()]
        algorithms_list = list(algorithms_info.keys())
        configurations_list = config_space_builder.build_config_space(search_spaces, defaults)
        print(configurations_list)

        # optimize the model with the default configurations and the configuration space
        meta_models_info = self.meta_model_utils.hyper_opt(x, y, algorithms_list,
                                                           configurations=configurations_list,
                                                           default_configurations=defaults,
                                                           time_budget=self.time_budget,
                                                           random_seed=self.random_seed)
        print(meta_models_info)
        # recommend the best algorithm
        meta_models_info_costs = [v['min_cost'] for v in meta_models_info.values()]
        print(f"min costs: {meta_models_info_costs}")
        print(f"best algorithm cost: {min(meta_models_info_costs)}")
        min_index = meta_models_info_costs.index(min(meta_models_info_costs))
        best_algorithm_name = algorithms_list[min_index]
        best_hyperparameters = meta_models_info[best_algorithm_name]['hyperparameters']
        trials_no = meta_models_info[best_algorithm_name]['trials_no']
        print(f"best selected algorithm: {best_algorithm_name}")
        print(f"best hyperparameters: {best_hyperparameters}")

        # evaluate best algorithm on train data

        start_time = time.time()
        training_results, fitted_modeling_pipeline, train_data_with_predictions \
            = algorithms_recommender.evaluate_algorithm(df=processed_train_data,
                                                        series_type=series_type,
                                                        algorithm_name=best_algorithm_name,
                                                        hyperparameters=best_hyperparameters)
        end_time = time.time()
        print("......... Evaluate the pipeline on train data .............")

        # evaluate best algorithm on test data
        if self.is_forecast:
            features_extraction_stage_index = -1
            lagged_stage_index = -1
            lagged_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[lagged_stage_index]
            last_significant_lags_index = lagged_stage.col_lags_dic[self.target_col_name]['last_significant_lag_index']
            significant_lags_no = last_significant_lags_index

            # get trend type
            trend_stage_index = 1
            trend_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[trend_stage_index]
            trend_type = trend_stage.trend_type

            # get no of seasonality components
            seasonality_stage_index = 2
            seasonality_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[
                seasonality_stage_index]
            seasonality_components_no = len(seasonality_stage.get_peak_frequencies())

            # concat lags in test data
            test_lagged_processor = LaggedDataPreprocessor(train_data=train_data, test_data=test_data,
                                                           num_lags=last_significant_lags_index)
            test_data = test_lagged_processor.preprocess_data()
        else:
            trend_type = None
            seasonality_components_no = None
            significant_lags_no = None

        evaluator = Evaluator(MLTasksEnum.REGRESSION)

        use_case_predictor = PredictorFactory(multi_step_flag=self.multi_step_flag,
                                              utils=self.utils,
                                              evaluator=evaluator,
                                              evaluation_metric_enum=self.evaluation_metric_enum,
                                              timestamp_col_name=self.time_stamp_col_name,
                                              target_col_name=self.target_col_name,
                                              prediction_col_name=self.prediction_col_name,
                                              processing_pipeline=fitted_processing_pipeline,
                                              modeling_pipeline=fitted_modeling_pipeline)

        predictor = use_case_predictor.create_predictor(train_data=train_data,
                                                        test_data=test_data,
                                                        num_lags=significant_lags_no)
        test_data_with_predictions, test_result = predictor.predict()

        # save train and test data with predictions
        training_result = {'params': best_hyperparameters,
                           'train_MAPE': training_results['MAPE'],
                           'test_MAPE': test_result['MAPE'],
                           'train_MSE': training_results['MSE'],
                           'test_MSE': test_result['MSE'],
                           'train_MAE': training_results['MAE'],
                           'test_MAE': test_result['MAE'],
                           'train_r2': training_results['R2'],
                           'test_r2': test_result['R2'],
                           'fitting_duration': end_time - start_time,
                           'series_type': series_type[self.target_col_name],
                           'trend_type': trend_type,
                           'seasonality_components_no': seasonality_components_no,
                           'lags_no': significant_lags_no,
                           'best_model': AlgorithmsMappingEnum[best_algorithm_name].value(**best_hyperparameters)
                           }
        # save train and test data
        train_data_with_predictions.to_csv(f'{self.results_path}/train_{self.dataset_name}')
        test_data_with_predictions.to_csv(f'{self.results_path}/test_{self.dataset_name}')

        return training_result
