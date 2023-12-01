import ast
import os
import time
import traceback
from math import ceil

import pandas as pd
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign

from GSAutoML.controller.meta_features_controller import MetaFeaturesController
from GSAutoML.controller.forecasting_controller import ForecastingController
from GSAutoML.data_modeling.regressors import Regressors
from GSAutoML.data_resampler.data_constructor import DataConstructor
from GSAutoML.enums.ML_tasks_enum import MLTasksEnum
from GSAutoML.enums.aggregations_enums import AggregationsEnum
from GSAutoML.enums.algorithms_mapping import AlgorithmsMappingEnum
from GSAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GSAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GSAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GSAutoML.enums.stages_enum import StagesEnum
from GSAutoML.meta_model.metafeatures_comparison.calculate_distance import DistanceCalculator
from GSAutoML.meta_model.optimization.Generate_initial_search_space import ConfigSpaceBuilder
from GSAutoML.meta_model.optimization.tune_hyperparameters import HyperparameterOptimizerBO
from GSAutoML.meta_model.top_algorithms import TopPerformers
from GSAutoML.pipelines.preprocessing_pipeline import PreprocessingPipeline
from GSAutoML.split_data.split_data import TimeSeriesSplitter


class MetaModelController:
    def __init__(self, dataset_instance, dataframe, meta_features_df, sorting_metric,
                 algorithms_performance_df, time_budget, not_all_features_flag=None):
        self.dataset_instance = dataset_instance
        self.dataset_name = self.dataset_instance.name
        self.dataframe = dataframe
        self.meta_features_df = meta_features_df
        self.algorithms_performance_df = algorithms_performance_df
        self.sorting_metric = sorting_metric
        self.time_budget = time_budget
        self.not_all_features_flag = not_all_features_flag
        self.exceptions = []

    @staticmethod
    def get_meta_features_real(df, dataset_instance):
        print("------------------ meta features of Real datasets for benchmarking ------------------------")
        date_col = 'Timestamp'
        target_col = 'Target'
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df.rename(columns={df.columns[0]: target_col})
        df.reset_index(inplace=True)
        # sort values by date_col
        df.sort_values(by=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)
        # resample dataset
        constructor = DataConstructor(date_col=date_col, target_col=target_col)
        resampled_df = constructor.resample(dataframe=df, agg_func=AggregationsEnum.AVG)
        print(resampled_df.shape)
        controller = MetaFeaturesController(resampled_df, dataset_instance)
        # todo: save calculated features to metafeatures table
        return controller.meta_features, resampled_df

    def run(self):
        print(f"=============  Start Meta Model Training for {self.dataset_name} ========================")

        # calculate meta features for the  new dataset
        print(self.dataframe)

        # run meta features controller on dataset
        new_meta_features, resampled_df = self.get_meta_features_real(self.dataframe,
                                                                      self.dataset_instance)
        new_meta_features_df = pd.DataFrame([new_meta_features])
        # append meta features of new dataset to all meta features
        print(self.meta_features_df.info())
        self.meta_features_df = self.meta_features_df.append(new_meta_features_df,
                                                             ignore_index=True)

        self.meta_features_df.drop(['features_no', 'dataset_ratio'], axis=1, inplace=True)
        if self.not_all_features_flag:
            # drop time series related features from meta features
            # todo: update pacf cloumns names
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

        print(self.meta_features_df['dataset_name'])

        # set dataset_name as index col # todo: exclude the dataset_name col instead
        self.meta_features_df.set_index('dataset_name', inplace=True)
        # initialize preprocessing pipeline of meta features
        processing_pipeline = PreprocessingPipeline(stages=[StagesEnum.Encoder.name,
                                                            StagesEnum.NORMALIZER.name])
        processing_pipeline.fit(self.meta_features_df)
        normalized_meta_features = processing_pipeline.transform(self.meta_features_df)
        print(normalized_meta_features.info())
        normalized_meta_features.reset_index(inplace=True)
        print(normalized_meta_features)

        # distance calculation - return top 1 closest dataset names
        distance_calc = DistanceCalculator(normalized_meta_features)
        closest_datasets = distance_calc.calculate_distances(query_dataset_id=self.dataset_name,
                                                             dataset_id_column='dataset_name',
                                                             top=3)
        print(closest_datasets)
        closest_dataset_name = closest_datasets['dataset_name'].iloc[0]
        print(f'Closest dataset: {closest_datasets}')

        # get algorithms configurations trained on the closest dataset
        performers = TopPerformers(self.algorithms_performance_df)

        # select top performer algorithms
        top_performers = performers.get_top_performers(closest_dataset_name=closest_dataset_name,
                                                       datasets_names_column='dataset_name')

        print(top_performers)

        # evaluate the algorithms on new dataset
        top_algorithms = top_performers[['algorithm', 'hyperparameters']]
        # exclude Gaussian if dataset size is larger than 10000 instance
        if resampled_df.shape[0] > 10000:
            if 'GaussianProcessRegressor' in list(top_algorithms['algorithm'].unique()):
                top_algorithms = top_algorithms[top_algorithms['algorithm'] != 'GaussianProcessRegressor']
        print(top_algorithms)
        # apply to forecasting controller
        # loop through selected algorithms
        top_performers_results_df = pd.DataFrame(
            columns=['algorithm', 'hyperparameters', 'MAPE', 'MSE',
                     'MAE', 'R2', 'fitting_duration'])

        for i, row in top_algorithms.iterrows():

            # todo: pass processed data directly and only train models here

            # create directory for saving results in for each algorithm
            # todo: handle meta model without case
            folder_path = f"AutoML/datasets/without_lasso_results/{row['algorithm']}"
            os.makedirs(folder_path, exist_ok=True)
            print(f"************** start training for {row['algorithm']} on new dataset *******************")
            print(row['hyperparameters'])
            print(type(row['hyperparameters']))
            # update hyperparameters format of specific algorithms
            if row['algorithm'] == RegressionAlgorithmsEnum.LightgbmRegressor.name:
                row['hyperparameters'] = ast.literal_eval(row['hyperparameters'])
                if row['hyperparameters'].get("max_depth", None) == -1:
                    row['hyperparameters']['max_depth'] = 9
                # drop verbose
                if 'verbose' in row['hyperparameters'].keys():
                    del row['hyperparameters']['verbose']
                parameters = row['hyperparameters']
            elif row['algorithm'] == RegressionAlgorithmsEnum.ElasticNetRegressor.name:
                # todo:handle elastic net issues (double check)
                row['hyperparameters'] = ast.literal_eval(row['hyperparameters'])
                if row['hyperparameters'].get("alpha", None) == 0:
                    row['hyperparameters']['alpha'] = 0.001
                parameters = row['hyperparameters']
            elif row['algorithm'] == RegressionAlgorithmsEnum.XGBoostRegressor.name:
                row['hyperparameters'] = ast.literal_eval(row['hyperparameters'])
                if row['hyperparameters'].get("gamma", None) < 0.9:
                    row['hyperparameters']['gamma'] = 0.9
                parameters = row['hyperparameters']
            else:
                parameters = ast.literal_eval(row['hyperparameters'])
            print(parameters)
            try:
                # train and evaluate these models with the same configuration on the closest dataset
                forecasting_controller = ForecastingController(
                    dataset_name=self.dataset_name,
                    dataframe=resampled_df,
                    algorithm_name=RegressionAlgorithmsEnum[row['algorithm']],
                    scoring_metric=RegressionScoringMetricEnum.MAE,
                    evaluation_metric_enum=RegressionEvaluationMetricEnum,
                    results_path=folder_path,
                    hyperparameters=parameters,
                    task_type=MLTasksEnum.REGRESSION)

                results, _ = forecasting_controller.train()
                print(results)
                # select only algorithm with hyperparameters and test results
                selected_results = {'algorithm': row['algorithm'],
                                    'hyperparameters': results['params'],
                                    'MAPE': results['test_MAPE'],
                                    'MSE': results['test_MSE'],
                                    'MAE': results['test_MAE'],
                                    'R2': results['test_r2'],
                                    'fitting_duration': results['fitting_duration']}
                # append results to rank them later
                top_performers_results_df = top_performers_results_df.append([selected_results])
                print(top_performers_results_df)
            except Exception as exc:
                print(f"found exception while training {row['algorithm']} on {self.dataset_name}")
                print(traceback.format_exc())
                self.exceptions.append({row['algorithm']: traceback.format_exc()})

        # select top 1 performer algorithms
        # sort by sorting metric to select top1 ---> sort by test score
        print(top_performers)
        print("algorithms before sorting")
        print(top_performers_results_df[['algorithm', self.sorting_metric]])
        print(f"sorting metric: {self.sorting_metric}=========================")
        if self.sorting_metric == 'R2':
            # sort descending for r2
            top_performers_results_df = top_performers_results_df.sort_values(self.sorting_metric,
                                                                              ascending=False).reset_index()
        else:
            # sort ascending for the rest of metrics
            top_performers_results_df = top_performers_results_df.sort_values(
                self.sorting_metric).reset_index()

        print(top_performers_results_df)
        print(top_performers_results_df[['algorithm', self.sorting_metric]])
        # selected_algorithm with hyperparameters
        print("================== Top performer =========================")

        # select top 3 performers
        top_3_algorithms_names = list(top_performers_results_df.iloc[:3]['algorithm'])
        print(top_3_algorithms_names)

        # split return x,y
        train_data, test_data = TimeSeriesSplitter(value_column='Target',
                                                   timestamp_column='Timestamp').split_data(data=resampled_df)

        print("============= Optimization on train data using SMAC ======================")
        # get search space from grid search algorithm
        # get hyperparameters as defaults
        algorithms_info = {}
        for algorithm in top_3_algorithms_names:
            best_algorithm_instance = top_performers_results_df[
                top_performers_results_df['algorithm'] == algorithm]

            # best_algorithm_name = top_performers_results_df['algorithm']
            best_algorithm = RegressionAlgorithmsEnum[algorithm]
            regressor = Regressors(regressor_name=best_algorithm, label_col_name="Target",
                                   time_stamp_col_name="Timestamp")
            search_space = regressor.get_regressor_params()
            defaults = best_algorithm_instance['hyperparameters'].iloc[0]
            print(defaults)

            algorithms_info[algorithm] = {'algorithm_instance': regressor,
                                          'search_space': search_space,
                                          'defaults': defaults}
        print(algorithms_info)

        # get processed data
        controller = ForecastingController(
            dataset_name=self.dataset_name,
            dataframe=resampled_df,
            algorithm_name=RegressionAlgorithmsEnum[top_3_algorithms_names[0]],
            scoring_metric=RegressionScoringMetricEnum.MAE,
            evaluation_metric_enum=RegressionEvaluationMetricEnum,
            results_path='',
            hyperparameters=list(algorithms_info.values())[0]['defaults'],
            training_flag=True,
            task_type=MLTasksEnum.REGRESSION)

        # get processed df of training data only
        processed_df = controller.train()
        print(processed_df.info())
        if RegressionAlgorithmsEnum.Adaboost.name in algorithms_info.keys():
            print(algorithms_info[RegressionAlgorithmsEnum.Adaboost.name])
            del algorithms_info[RegressionAlgorithmsEnum.Adaboost.name]['defaults']['estimator']
        elif RegressionAlgorithmsEnum.GaussianProcessRegressor.name in algorithms_info.keys():
            algorithms_info[RegressionAlgorithmsEnum.GaussianProcessRegressor.name]['defaults']['sigma_0'] \
                = algorithms_info[RegressionAlgorithmsEnum.GaussianProcessRegressor.name]['defaults'].get("kernel",
                                                                                                          {}).get(
                "sigma_0", None)
            del algorithms_info[RegressionAlgorithmsEnum.GaussianProcessRegressor.name]['defaults']['kernel']

        processed_df.set_index('Timestamp', inplace=True)
        y = processed_df['Target']
        x = processed_df.drop('Target', axis=1)

        # build configuration space for the selected algorithm

        config_space_builder = ConfigSpaceBuilder()
        # list of config spaces
        search_spaces = [v['search_space'] for v in algorithms_info.values()]
        defaults = [v['defaults'] for v in algorithms_info.values()]
        algorithms = list(algorithms_info.keys())
        configurations_list = config_space_builder.build_config_space(search_spaces, defaults)
        # divide time budget
        # optimize the model with the default configurations and the configuration space
        time_budgets = ceil(self.time_budget / len(algorithms))

        meta_models_info = {}
        for i, cs in enumerate(configurations_list):
            optimizer = HyperparameterOptimizerBO(cs, x, y,
                                                  algorithm_name=
                                                  AlgorithmsMappingEnum[algorithms[i]].value)

            scenario = Scenario(cs, n_trials=10000000, walltime_limit=time_budgets * 60)
            initial_design = DefaultInitialDesign(scenario)
            smac = HyperparameterOptimizationFacade(scenario, optimizer.train,
                                                    initial_design=initial_design,
                                                    overwrite=True)

            start = time.time()
            incumbent = smac.optimize()
            print("time:", time.time() - start)
            # Access the run history from the SMAC object

            runhistory = smac.runhistory.get_configs(sort_by='cost')
            print(runhistory)
            best_params = runhistory[0]
            print(best_params)
            min_cost = smac.runhistory.get_cost(best_params)
            print(f"min cost:,{min_cost}")

            run_id = smac.runhistory.get_config_id(best_params)
            print(run_id)

            best_hyperparameters = dict(best_params)
            print(f"best params:{dict(best_params)}")
            meta_models_info[algorithms[i]] = {'min_cost': min_cost,
                                               'hyperparameters': best_hyperparameters,
                                               'time': time.time() - start,
                                               'default': defaults[i],
                                               'trial_1': smac.runhistory.get_configs()[0],
                                               'trials_no': len(runhistory)}

        print(meta_models_info)
        # return best algorithm out of the 3 meta-models
        meta_models_info_costs = [v['min_cost'] for v in meta_models_info.values()]
        print(f"min costs: {meta_models_info_costs}")
        print(f"best algorithm cost: {min(meta_models_info_costs)}")
        min_index = meta_models_info_costs.index(min(meta_models_info_costs))
        best_algorithm_name = algorithms[min_index]
        best_hyperparameters = meta_models_info[best_algorithm_name]['hyperparameters']
        trials_no = meta_models_info[best_algorithm_name]['trials_no']
        print(f"best selected algorithm: {best_algorithm_name}")
        print(f"best hyperparameters: {best_hyperparameters}")

        # updates for hyperparameters format for specific algorithms
        if best_algorithm_name == RegressionAlgorithmsEnum.GaussianProcessRegressor.name:
            best_hyperparameters['kernel'] = {"Type": "DotProduct",
                                              'sigma_0': best_hyperparameters.get('sigma_0', None)}
            if 'sigma_0' in best_hyperparameters.keys():
                del best_hyperparameters['sigma_0']
        elif best_algorithm_name == RegressionAlgorithmsEnum.Adaboost.name:
            best_hyperparameters['estimator'] = {"Type": "DecisionTreeRegressor", "max_depth": 5}

        print(best_hyperparameters)
        print("================  Train and evaluate best model on data ================================")
        # create a directory to save results of the best algorithms selected by meta model
        # todo: create one for without case
        # AutoML/datasets/meta_model_without_results/final_{self.time_budget}_min
        path = f"AutoML/datasets/without_lasso_results/final_{self.time_budget}_min"
        os.makedirs(path, exist_ok=True)
        final_forecasting_controller = ForecastingController(
            dataset_name=best_algorithm_name + '_' + self.dataset_name,
            dataframe=resampled_df,
            algorithm_name=RegressionAlgorithmsEnum[best_algorithm_name],
            scoring_metric=RegressionScoringMetricEnum.MAE,
            evaluation_metric_enum=RegressionEvaluationMetricEnum,
            results_path=path,
            hyperparameters=best_hyperparameters,
            task_type=MLTasksEnum.REGRESSION)
        evaluation_results, _ = final_forecasting_controller.train()

        print(evaluation_results)

        evaluation_results['best_model'] = \
            AlgorithmsMappingEnum[best_algorithm_name].value(**evaluation_results['params'])

        meta_model_results = {
            'meta_model_trials': meta_models_info,
            'nearest_dataset_name': closest_dataset_name,
            'closest_distance': closest_datasets['distance'].iloc[0],
            'top_3_nearest_datasets': closest_datasets.iloc[:3],  # save as df :(
            'trials_no': trials_no,
            'default_algorithm': best_algorithm_name,
            'default_hyperparameters': meta_models_info[best_algorithm_name]['default'],
            'normalized_meta_features':
                normalized_meta_features[normalized_meta_features['dataset_name'] == self.dataset_name],
            'nearest_normalized_meta_features':
                normalized_meta_features[normalized_meta_features['dataset_name'] == closest_dataset_name]
        }
        print(meta_model_results)

        return evaluation_results, meta_model_results
        # todo: how to evaluate meta model?
