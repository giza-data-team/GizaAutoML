from GSAutoML.data_modeling.evaluator import Evaluator
from GSAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GSAutoML.controller.common.utils import Utils
from GSAutoML.enums.automl_engines_enum import AutomlEnginesEnum
from GSAutoML.multistep_forecasting.multistep_forecasting import MultistepTimeSeriesForecaster
from GSAutoML.split_data.split_data import TimeSeriesSplitter
import time
from GSAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GSAutoML.pipelines.automl_pipeline import AutomlPipeline
from GSAutoML.enums.ML_tasks_enum import MLTasksEnum
from GSAutoML.controller.common.use_case_predictor_factory import PredictorFactory


class AutomlController:
    """ A controller for pre-built Auto ML engines"""

    def __init__(self, dataframe, dataset_name, results_path, engine_name: AutomlEnginesEnum,
                 scoring_metric,
                 time_budget: int, random_seed: int, multi_step_flag: bool = False):
        self.dataframe = dataframe
        self.engine_name = engine_name
        self.scoring_metric = scoring_metric
        self.time_budget = time_budget
        self.dataset_name = dataset_name
        self.results_path = results_path
        self.target_col_name = 'Target'
        self.prediction_col_name = 'prediction'
        self.time_stamp_col_name = 'Timestamp'
        self.random_seed = random_seed
        self.utils = Utils()
        self.evaluation_metric_enum = RegressionEvaluationMetricEnum
        self.multi_step_flag = multi_step_flag

    def train(self):

        # split data to train and test
        train_data, test_data = TimeSeriesSplitter(value_column=self.target_col_name,
                                                   timestamp_column=self.time_stamp_col_name).split_data(
            data=self.dataframe)
        series_type = self.utils.get_series_type(train_data)
        processed_data, fitted_processing_pipeline = self.utils.get_processed_data(train_data)
        # initialize auto regression engine
        automl_pipeline = AutomlPipeline(label_col=self.target_col_name,
                                         prediction_col=self.prediction_col_name,
                                         scoring_metric=self.scoring_metric,
                                         engine=self.engine_name,
                                         exclude_cols=[self.time_stamp_col_name],
                                         time_stamp_col_name=self.time_stamp_col_name,
                                         seasonality_mode=series_type,
                                         time_budget=self.time_budget,
                                         random_seed=self.random_seed)

        print("......... Fit and Transform the AutoRegressor pipeline on training data .............")

        # calculate engine execution time
        start_time = time.time()
        fitted_automl_pipeline = automl_pipeline.fit(X=processed_data)
        end_time = time.time()

        train_data_with_predictions = fitted_automl_pipeline.transform(X=processed_data)

        print("......... Evaluate the pipeline on train data.............")
        # evaluate model performance on training data
        evaluator = Evaluator(MLTasksEnum.REGRESSION)
        train_results = evaluator.evaluate(evaluation_metric_enum=self.evaluation_metric_enum,
                                           actual_data=train_data_with_predictions[self.target_col_name],
                                           predicted_data=train_data_with_predictions[self.prediction_col_name])

        print("......... Transform the pipeline on test data .............")

        # get no of significant lags of training data
        features_extraction_stage_index = 1

        lagged_stage_index = -1
        lagged_stage = fitted_processing_pipeline[features_extraction_stage_index].stages[lagged_stage_index]
        significant_lags_no = lagged_stage.col_lags_dic[self.target_col_name]['last_significant_lag_index']
        use_case_predictor = PredictorFactory(multi_step_flag=self.multi_step_flag,
                                              utils=self.utils,
                                              evaluator=evaluator,
                                              evaluation_metric_enum=self.evaluation_metric_enum,
                                              timestamp_col_name=self.time_stamp_col_name,
                                              target_col_name=self.target_col_name,
                                              prediction_col_name=self.prediction_col_name,
                                              processing_pipeline=fitted_processing_pipeline,
                                              modeling_pipeline=fitted_automl_pipeline)

        predictor = use_case_predictor.create_predictor(train_data=train_data,
                                                        test_data=test_data,
                                                        num_lags=significant_lags_no)
        test_data_with_predictions, test_results = predictor.predict()

        if self.engine_name == AutomlEnginesEnum.tpot:
            best_model = fitted_automl_pipeline.steps[0][1].estimator.fitted_pipeline_
        elif self.engine_name == AutomlEnginesEnum.auto_sklearn:

            best_model = list(fitted_automl_pipeline.steps[0][1].estimator.show_models().values())[0].\
            get('estimators')[0].get('sklearn_regressor')
            print(best_model)
        evaluation_results = {}

        for key, value in train_results.items():
            # Append "train_" prefix to the key and store the value
            evaluation_results["train_" + key] = value
        # Iterate over the keys in the second dictionary
        for key, value in test_results.items():
            # Append "test_" prefix to the key and store the value
            evaluation_results["test_" + key] = value

        training_result = {'best_model': best_model,
                           'fitting_duration': end_time - start_time,
                           'series_type': series_type['Target'],
                           **evaluation_results

                           }

        print(training_result)
        # save train and test results
        train_data_with_predictions.to_csv(f'{self.results_path}/train_{self.dataset_name}')
        test_data_with_predictions.to_csv(f'{self.results_path}/test_{self.dataset_name}')

        return training_result
