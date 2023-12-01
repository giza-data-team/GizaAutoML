import ast
import re
import traceback

import pandas as pd

from GSAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GSAutoML.controller.common.utils import Utils
from GSAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GSAutoML.enums.stages_enum import StagesEnum
from GSAutoML.enums.ML_tasks_enum import MLTasksEnum
from GSAutoML.meta_model.metafeatures_comparison.calculate_distance import DistanceCalculator
from GSAutoML.meta_model.top_algorithms import TopPerformers
from GSAutoML.pipelines.ML_pipeline_factory import MLFactory
from GSAutoML.pipelines.preprocessing_pipeline import PreprocessingPipeline
from GSAutoML.data_modeling.evaluator import Evaluator
from GSAutoML.controller.common.knowledge_base_collector import KnowledgeBaseCollector


class AlgorithmsRecommender:
    """
       class for recommending regression algorithms based on meta features and performance data.

       This class provides functionality to recommend the top-performing regression algorithms
       for a given dataset based on its meta features and past performance data.

       Args:
           series_name (str): The name of the dataset or time series saved in datasets table.
           series (DataFrame): The dataset or time series data as a DataFrame.
           series_meta_features (dict): Meta features extracted from the dataset.
           sorting_metric (str): The metric used for sorting and ranking the algorithms.

       Attributes:
           exceptions (list): A list to store any exceptions encountered during processing.
           series (DataFrame): The dataset or time series data.
           series_meta_features (dict): Meta features extracted from the dataset.
           dataset_name (str): The name of the dataset or time series.
           meta_features_df (DataFrame): A DataFrame containing meta features for multiple datasets.
           algorithms_performance_df (DataFrame): A DataFrame containing performance data for regression algorithms.
           sorting_metric (str): The metric used for sorting and ranking algorithms.
           not_all_features_flag (bool): A flag indicating whether not all features are included.
           utils (Utils): An instance of the utility class for common operations.
           target_col_name (str): The name of the target column in the dataset.
           prediction_col_name (str): The name of the column for storing algorithm predictions.
           time_stamp_col_name (str): The name of the timestamp column if applicable.
           normalized_meta_features (DataFrame): A DataFrame containing normalized meta features.
           closest_3_datasets (DataFrame): A DataFrame containing information about the three closest datasets
                                           based on meta features.
       """

    def __init__(self, series_name, series, series_meta_features, sorting_metric, target_col='Target'):
        self.exceptions = []
        self.series = series
        self.series_meta_features = series_meta_features
        self.dataset_name = series_name
        self.meta_features_df, self.algorithms_performance_df = self._get_kb()
        self.sorting_metric = sorting_metric
        self.not_all_features_flag = False
        self.utils = Utils(series_col=target_col)
        self.target_col_name = target_col
        self.prediction_col_name = 'prediction'
        self.time_stamp_col_name = 'Timestamp'
        # todo: enhance
        self.normalized_meta_features = None
        self.closest_3_datasets = None

    @staticmethod
    def _get_kb():
        """ get the dataframes of the meta features and algorithms performance from the KB """
        print("===================== Access KB ==============================")
        # KB collection
        KB_collector = KnowledgeBaseCollector()
        regression_algorithms_df = KB_collector.get_regression_algorithms_performance()
        meta_features_df = KB_collector.get_meta_features()
        return meta_features_df, regression_algorithms_df

    def get_closest_dataset(self):
        """
        Retrieves information about the three closest datasets based on meta features.

        This method appends the meta features of a new dataset to the existing meta features dataframe,
        processes and normalizes the data, calculates the distances between datasets, and returns the
        top three closest dataset names.

        Returns:
            closest_datasets (DataFrame): A DataFrame containing information about the three closest datasets
                                          based on their meta features.
        """
        new_meta_features_df = pd.DataFrame([self.series_meta_features])
        # append meta features of new dataset to all meta features
        self.meta_features_df = self.meta_features_df.append(new_meta_features_df,
                                                             ignore_index=True)

        if self.not_all_features_flag:
            # drop time series related features from meta features
            columns_to_drop = ['sampling_rate', 'stationary_no', 'non_stationary_no',
                               'first_diff_stationary_no', 'second_diff_stationary_no',
                               'lags_no', 'seasonality_components_no', 'fractal_dim',
                               'series_type', 'trend_type', 'insignificant_lags_no']
            for i in range(1, 11):
                if i != 10:
                    columns_to_drop.append(f"pacf_0{i}")
                else:
                    columns_to_drop.append(f"pacf_{i}")
            print(columns_to_drop)
            self.meta_features_df.drop(columns_to_drop,
                                       axis=1, inplace=True)

        # set dataset_name as index col
        self.meta_features_df.set_index('dataset_name', inplace=True)
        # initialize preprocessing pipeline of meta features
        processing_pipeline = PreprocessingPipeline(stages=[StagesEnum.Encoder.name,
                                                            StagesEnum.NORMALIZER.name])
        processing_pipeline.fit(self.meta_features_df)
        self.normalized_meta_features = processing_pipeline.transform(self.meta_features_df)
        self.normalized_meta_features.reset_index(inplace=True)

        # distance calculation - return top 3 closest dataset names
        distance_calc = DistanceCalculator(self.normalized_meta_features)
        closest_datasets = distance_calc.calculate_distances(query_dataset_id=self.dataset_name,
                                                             dataset_id_column='dataset_name',
                                                             top=3)
        self.closest_3_datasets = closest_datasets

        return closest_datasets

    def get_algorithms_configuration(self, closest_dataset_name):
        """
        Retrieves the configurations of top-performing algorithms trained on the closest dataset.

        This method utilizes performance data and selects the algorithms with the best performance
        on the specified closest dataset. It returns a DataFrame containing information about the
        selected algorithms and their hyperparameters.

        Args:
            closest_dataset_name (str): The name of the closest dataset for which algorithm configurations
                                        are to be retrieved.

        Returns:
            algorithms_config (DataFrame): A DataFrame containing algorithm names and their corresponding
                                           hyperparameters for the top-performing algorithms on the specified
                                           closest dataset. GaussianProcessRegressor is excluded if the dataset
                                           size is larger than 10,000 instances for memory issues.
        """

        # get algorithms configurations trained on the closest dataset
        performers = TopPerformers(self.algorithms_performance_df)
        # select top performer algorithms
        top_performers = performers.get_top_performers(closest_dataset_name=closest_dataset_name,
                                                       datasets_names_column='dataset_name')

        algorithms_config = top_performers[['algorithm', 'hyperparameters']]

        # exclude Gaussian if dataset size is larger than 10000 instance
        if self.series.shape[0] > 10000:
            if 'GaussianProcessRegressor' in list(algorithms_config['algorithm'].unique()):
                algorithms_config = algorithms_config[algorithms_config['algorithm'] != 'GaussianProcessRegressor']
        return algorithms_config

    def get_scores(self, df):
        """
        Calculates and returns various regression performance scores for model evaluation.

        This method calculates and returns the following regression performance scores based on the
        provided DataFrame 'df'.
        Args:
            df (DataFrame): A DataFrame containing actual and predicted values for regression analysis.

        Returns:
            results (dict): A dictionary containing the calculated regression performance scores.
                - 'MAPE' (float): Mean Absolute Percentage Error
                - 'MSE' (float): Mean Squared Error
                - 'MAE' (float): Mean Absolute Error
                - 'R2' (float): R-squared (coefficient of determination) score
        """
        evaluator = Evaluator(MLTasksEnum.REGRESSION)
        results = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum,
                                     actual_data=df[self.target_col_name],
                                     predicted_data=df[self.prediction_col_name])

        return results

    def evaluate_algorithm(self, df, series_type, algorithm_name, hyperparameters, target_col='Target'):
        """
        Evaluates a regression algorithm's performance on a given DataFrame.

        This method evaluates the performance of a regression algorithm on the provided DataFrame 'df'
        using the specified algorithm, hyperparameters, and series type. It returns the training results
        and the fitted modeling pipeline.

        Args:
            df (DataFrame): A DataFrame containing the dataset for regression analysis.
            series_type (str): The type of time series data ('additive' or 'multiplicative').
            algorithm_name (str): The name of the regression algorithm to evaluate.
            hyperparameters (dict): A dictionary of hyperparameters for the chosen algorithm.
            target_col (str): the target column name

        Returns:
            training_results (dict): A dictionary containing regression performance scores on the training data.
                - 'MAPE' (float): Mean Absolute Percentage Error
                - 'MSE' (float): Mean Squared Error
                - 'MAE' (float): Mean Absolute Error
                - 'R2' (float): R-squared (coefficient of determination) score
            fitted_modeling_pipeline (RegressorPipeline): A fitted regression modeling pipeline.
        """
        modeling_pipeline = MLFactory.create_pipeline(
            task_type=MLTasksEnum.REGRESSION,
            label_col=target_col,
            prediction_col='prediction',
            scoring_metric=self.sorting_metric,
            estimator=RegressionAlgorithmsEnum[algorithm_name],
            exclude_cols=[],
            time_stamp_col_name='Timestamp',
            seasonality_mode=series_type,
            hyperparameters=hyperparameters
        )

        fitted_modeling_pipeline = modeling_pipeline.fit(X=df)
        train_data_with_predictions = fitted_modeling_pipeline.transform(df)
        training_results = self.get_scores(train_data_with_predictions)

        return training_results, fitted_modeling_pipeline

    def get_top_algorithms(self):
        """
        Identifies and ranks the top-performing regression algorithms for the given dataset.

        This method performs the following steps:
        1. Obtains information about the closest dataset based on meta features.
        2. Retrieves configurations of top-performing algorithms trained on the closest dataset.
        3. Evaluates the selected algorithms on the current dataset, recording their performance.
        4. Ranks the algorithms based on the specified sorting metric (e.g., R-squared, MAE).
        5. Returns the top three performing algorithms along with their performance metrics.

        Returns:
            top_performers_results_df (DataFrame): A DataFrame containing information about the top three
                performing regression algorithms, including algorithm names, hyperparameters, and their
                performance scores.
        """
        # get closest dataset name
        closest_datasets = self.get_closest_dataset()
        closest_dataset_name = closest_datasets['dataset_name'].iloc[0]
        print(f'Closest dataset: {closest_datasets}')

        # get algorithms configurations trained on the closest dataset
        algorithms_config = self.get_algorithms_configuration(closest_dataset_name)

        # evaluate the algorithms on new dataset
        series_type = self.utils.get_series_type(self.series)
        processed_data, fitted_processing_pipeline = self.utils.get_processed_data(self.series)

        # loop through selected algorithms and evaluate them
        top_performers_results_df = pd.DataFrame(
            columns=['algorithm', 'hyperparameters', 'MAPE', 'MSE',
                     'MAE', 'R2'])

        for i, row in algorithms_config.iterrows():

            print(f"************** start training for {row['algorithm']} on new dataset *******************")
            # update hyperparameters format of specific algorithms
            if row['algorithm'] == RegressionAlgorithmsEnum.AdaboostRegressor.name:
                # Define a regular expression pattern to match the 'estimator' key-value pair
                parameters = row['hyperparameters']
                pattern = r"'estimator':[^,}]+,? ?"
                # Find the 'estimator' key-value pair
                parameters = re.sub(pattern, '', parameters)

                # Define a regular expression pattern to capture the 'max_depth' value
                max_depth_pattern = r'max_depth=(\d+)'

                # Use re.search to find and capture the 'max_depth' value
                max_depth_match = re.search(max_depth_pattern, row['hyperparameters'])
                if max_depth_match:
                    max_depth_value = int(max_depth_match.group(1))  # Convert to an integer

                parameters = ast.literal_eval(parameters)
                parameters['max_depth'] = max_depth_value

            elif row['algorithm'] == RegressionAlgorithmsEnum.GaussianProcessRegressor.name:
                # todo: handle gaussian kernel parsing issue
                continue

            else:
                parameters = ast.literal_eval(row['hyperparameters'])
            try:
                # train and evaluate these models with the same configuration on the closest dataset

                results, fitted_modeling_pipeline = self.evaluate_algorithm(processed_data,
                                                                            series_type,
                                                                            algorithm_name=row['algorithm'],
                                                                            hyperparameters=parameters)
                # select only algorithm with hyperparameters and test results
                selected_results = {'algorithm': row['algorithm'],
                                    'hyperparameters': parameters,
                                    'MAPE': results['MAPE'],
                                    'MSE': results['MSE'],
                                    'MAE': results['MAE'],
                                    'R2': results['R2'],
                                    }
                # append results to rank them later
                top_performers_results_df = top_performers_results_df.append([selected_results])
                print(top_performers_results_df)
            except Exception as exc:
                print(f"found exception while training {row['algorithm']} on {self.dataset_name}")
                print(traceback.format_exc())
                self.exceptions.append({row['algorithm']: traceback.format_exc()})

        # sort by sorting metric to select top 3
        print(f"sorting metric: {self.sorting_metric}=========================")
        if self.sorting_metric == 'R2':
            # sort descending for r2
            top_performers_results_df = top_performers_results_df.sort_values(self.sorting_metric,
                                                                              ascending=False).reset_index()
        else:
            # sort ascending for the rest of metrics
            top_performers_results_df = top_performers_results_df.sort_values(
                self.sorting_metric).reset_index()

        print(top_performers_results_df[['algorithm', self.sorting_metric]])

        return top_performers_results_df.iloc[:3]
