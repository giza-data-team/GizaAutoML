import time

from GizaAutoML.controller.common.utils import Utils
from GizaAutoML.data_modeling.evaluator import Evaluator
from GizaAutoML.enums.ML_tasks_enum import MLTasksEnum
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GizaAutoML.pipelines.ML_pipeline_factory import MLFactory
from GizaAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GizaAutoML.split_data.split_data import TimeSeriesSplitter


class ForecastingController:
    def __init__(self, dataset_name, dataframe, algorithm_name, results_path, evaluation_metric_enum,
                 scoring_metric, timestamp_col, series_col, target_labels_col, prediction_labels_col,
                 is_forecast=True, hyperparameters=None, task_type=MLTasksEnum.REGRESSION):
        self.dataset_name = dataset_name
        self.dataframe = dataframe
        self.algorithm_name = algorithm_name
        self.scoring_metric = scoring_metric
        self.results_path = results_path
        self.is_forecast = is_forecast
        self.hyperparameters = hyperparameters
        self.prediction_col_name = prediction_labels_col
        self.time_stamp_col_name = timestamp_col
        self.task_type = task_type
        self.target_col_name = target_labels_col if self.task_type == MLTasksEnum.CLASSIFICATION else 'Target'
        self.series_name = series_col
        self.evaluation_metric_enum = evaluation_metric_enum
        self.utils = Utils(series_col=series_col, timestamp_col_name=timestamp_col,
                           columns_to_exclude=[timestamp_col, self.target_col_name],
                           is_forecast=self.is_forecast)

    def train(self):
        print("--------------------  IN forecasting controller ----------------------------")
        # split data to train and test
        print(self.dataframe)
        train_data, test_data = (TimeSeriesSplitter(value_column=self.series_name,
                                                    timestamp_column=self.time_stamp_col_name).
                                 split_data(data=self.dataframe))

        series_type = self.utils.get_series_type(train_data)
        processed_data, fitted_processing_pipeline = self.utils.get_processed_data(train_data)
        # initialize classical regression modeling pipeline
        modeling_pipeline = MLFactory.create_pipeline(task_type=self.task_type,
                                                      label_col=self.target_col_name,
                                                      prediction_col=self.prediction_col_name,
                                                      scoring_metric=self.scoring_metric,
                                                      estimator=self.algorithm_name,
                                                      exclude_cols=[],
                                                      time_stamp_col_name=self.time_stamp_col_name,
                                                      seasonality_mode=series_type,
                                                      hyperparameters=self.hyperparameters
                                                      )
        print("......... Fit and Transform the Forecasting pipeline on training data .............")

        # calculate grid search execution time
        start_time = time.time()
        fitted_modeling_pipeline = modeling_pipeline.fit(X=processed_data)
        end_time = time.time()

        # get grid search results
        grid_search_results = fitted_modeling_pipeline.steps[0][1].grid_search_results
        train_data_with_predictions = fitted_modeling_pipeline.transform(processed_data)
        # apply pipeline on test data

        print("......... Transform the pipeline on test data .............")
        parent_pipeline = self.utils.create_pipeline([fitted_processing_pipeline, fitted_modeling_pipeline])
        features_extraction_stage_index = 1

        if self.is_forecast:
            # get trend type
            trend_stage_index = 1
            trend_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[trend_stage_index]
            trend_type = trend_stage.trend_type

            # get no of seasonality components
            seasonality_stage_index = 2
            seasonality_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[seasonality_stage_index]
            seasonality_components_no = len(seasonality_stage.get_peak_frequencies())
            # get no of significant lags of training data
            lagged_stage_index = -1
            lagged_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[lagged_stage_index]
            last_significant_lags_index = lagged_stage.col_lags_dic[self.series_name]['last_significant_lag_index']
            significant_lags_no = last_significant_lags_index

            # concat lags in test data
            test_lagged_processor = LaggedDataPreprocessor(train_data=train_data, test_data=test_data,
                                                           num_lags=last_significant_lags_index)
            test_data = test_lagged_processor.preprocess_data()
        else:
            trend_type = None
            seasonality_components_no = None
            significant_lags_no = None



        # transforming on test data
        test_data_with_predictions = parent_pipeline.transform(test_data)
        print("......... Evaluate the pipeline on train and test data .............")

        # evaluate model performance on training data
        evaluator = Evaluator(self.task_type)
        train_results = evaluator.evaluate(evaluation_metric_enum=self.evaluation_metric_enum,
                                           actual_data=train_data_with_predictions[self.target_col_name],
                                           predicted_data=train_data_with_predictions[self.prediction_col_name])

        # evaluate model performance on test data
        test_results = evaluator.evaluate(evaluation_metric_enum=self.evaluation_metric_enum,
                                          actual_data=test_data_with_predictions[self.target_col_name],
                                          predicted_data=test_data_with_predictions[self.prediction_col_name])

        if self.hyperparameters:
            best_params = self.hyperparameters
        else:
            best_params = modeling_pipeline.steps[0][1].best_params
        evaluation_results = {}

        for key, value in train_results.items():
            # Append "train_" prefix to the key and store the value
            evaluation_results["train_" + key] = value
        # Iterate over the keys in the second dictionary
        for key, value in test_results.items():
            # Append "test_" prefix to the key and store the value
            evaluation_results["test_" + key] = value

        training_result = {'params': best_params,
                           'fitting_duration': end_time - start_time,
                           'series_type': series_type[self.series_name],
                           'trend_type': trend_type,
                           'seasonality_components_no': seasonality_components_no,
                           'lags_no': significant_lags_no,
                           **evaluation_results
                           }
        print(training_result)
        # save train and test results
        train_data_with_predictions.to_csv(f'{self.results_path}/train_{self.dataset_name}')
        test_data_with_predictions.to_csv(f'{self.results_path}/test_{self.dataset_name}')

        return training_result, grid_search_results
