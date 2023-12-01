import sklearn
from GSAutoML.enums.ML_tasks_enum import MLTasksEnum
from GSAutoML.data_modeling.evaluator import Evaluator
from GSAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GSAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GSAutoML.enums.statistical_models_enum import StatisticalModelsEnum
from GSAutoML.pipelines.feature_extraction_pipeline import \
    FeatureExtractionPipeline
from GSAutoML.enums.stages_enum import StagesEnum
from GSAutoML.feature_engineering.data_preproccessing.estimators.series_type_estimator import \
    SeriesTypeEstimator
from sklearn.pipeline import Pipeline

from GSAutoML.pipelines.regression_pipeline import RegressorPipeline
from GSAutoML.split_data.split_data import TimeSeriesSplitter
import time
from GSAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GSAutoML.pipelines.preprocessing_pipeline import PreprocessingPipeline


class StatisticalController:
    def __init__(self, dataframe, algorithm_name: StatisticalModelsEnum, scoring_metric: RegressionScoringMetricEnum):
        self.dataframe = dataframe
        self.algorithm_name = algorithm_name
        self.scoring_metric = scoring_metric
        self.target_col_name = 'Target'
        self.prediction_col_name = 'yhat'
        self.time_stamp_col_name = 'Timestamp'

    @staticmethod
    def _create_pipeline(processing_pipeline_stages):
        """Create sklearn pipeline"""
        pp_steps = [(stage.__class__.__name__ + f"_{idx}", stage) for idx, stage in
                    enumerate(processing_pipeline_stages)]
        return Pipeline(steps=pp_steps)

    def train(self):
        pipelines = []
        # apply resampling

        # split data to train and test
        train_data, test_data = TimeSeriesSplitter(value_column='Target').split_data(data=self.dataframe)
        # check series type
        st_estimator = SeriesTypeEstimator()
        series_type = st_estimator.series_type_estimator(train_data)
        print(f"Series Type: {series_type}")

        # initialize preprocessing pipeline with imputer stage
        pipelines.append(PreprocessingPipeline(series_types=series_type, stages=[StagesEnum.IMPUTER.name]))

        # initialize feature extraction pipeline
        # feature_extraction_pipeline = FeatureExtractionPipeline(
        #     dataframe=train_data,
        #     stages=[StagesEnum.TREND.name, StagesEnum.SEASONALITY.name,
        #             StagesEnum.LAGGED.name, StagesEnum.TIME.name],
        #     series_types=series_type)

        preprocessing_pipeline = self._create_pipeline(pipelines)

        print("......... Fit and Transform the Preprocessing pipeline on training data .............")

        # fit on train data
        fitted_processing_pipeline = preprocessing_pipeline.fit(train_data)

        # predict on train data
        processed_data = fitted_processing_pipeline.transform(train_data)

        # initialize classical regression modeling pipeline
        modeling_pipeline = RegressorPipeline(label_col=self.target_col_name,
                                              prediction_col=self.prediction_col_name,
                                              scoring_metric=self.scoring_metric,
                                              estimator=self.algorithm_name,
                                              exclude_cols=[self.time_stamp_col_name],
                                              time_stamp_col_name=self.time_stamp_col_name,
                                              seasonality_mode=series_type,
                                              )

        print("......... Fit and Transform the Regressor pipeline on training data .............")

        # calculate grid search execution time
        start_time = time.time()
        fitted_modeling_pipeline = modeling_pipeline.fit(X=processed_data)
        end_time = time.time()

        # get grid search results
        grid_search_results = fitted_modeling_pipeline.steps[0][1].grid_search_results
        print(grid_search_results)
        train_data_with_predictions = fitted_modeling_pipeline.transform(processed_data)
        # apply pipeline on test data

        print("......... Transform the pipeline on test data .............")
        parent_pipeline = self._create_pipeline([fitted_processing_pipeline, fitted_modeling_pipeline])

        # get no of significant lags of training data
        # lagged_stage_index = 2
        # lagged_stage = fitted_processing_pipeline[-1].stages[lagged_stage_index]
        # significant_lags_no = lagged_stage.col_lags_dic[self.target_col_name]['last_significant_lag_index']

        # concat lags in test data
        test_lagged_processor = LaggedDataPreprocessor(train_data=train_data, test_data=test_data,
                                                       num_lags=20)
        test_data = test_lagged_processor.preprocess_data()

        # transforming on test data
        test_data_with_predictions = parent_pipeline.transform(test_data)
        print("......... Evaluate the pipeline on train and test data .............")

        # evaluate model performance on training data
        evaluator = Evaluator(MLTasksEnum.REGRESSION)
        train_MAPE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAPE,
                                        actual_data=train_data_with_predictions[self.target_col_name],
                                        predicted_data=train_data_with_predictions[self.prediction_col_name])

        train_MSE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MSE,
                                       actual_data=train_data_with_predictions[self.target_col_name],
                                       predicted_data=train_data_with_predictions[self.prediction_col_name])

        train_MAE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAE,
                                       actual_data=train_data_with_predictions[self.target_col_name],
                                       predicted_data=train_data_with_predictions[self.prediction_col_name])
        train_r2 = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.R2,
                                      actual_data=train_data_with_predictions[self.target_col_name],
                                      predicted_data=train_data_with_predictions[self.prediction_col_name])

        # evaluate model performance on test data
        test_MAPE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAPE,
                                       actual_data=test_data_with_predictions[self.target_col_name],
                                       predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_MSE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MSE,
                                      actual_data=test_data_with_predictions[self.target_col_name],
                                      predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_MAE = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.MAE,
                                      actual_data=test_data_with_predictions[self.target_col_name],
                                      predicted_data=test_data_with_predictions[self.prediction_col_name])
        test_r2 = evaluator.evaluate(evaluation_metric_enum=RegressionEvaluationMetricEnum.R2,
                                     actual_data=test_data_with_predictions[self.target_col_name],
                                     predicted_data=test_data_with_predictions[self.prediction_col_name])

        params = modeling_pipeline.steps[0][1].best_params
        best_params = params.copy()

        training_result = {'params': best_params,
                           'train_MAPE': train_MAPE,
                           'test_MAPE': test_MAPE,
                           'train_MSE': train_MSE,
                           'test_MSE': test_MSE,
                           'train_MAE': train_MAE,
                           'test_MAE': test_MAE,
                           'train_r2': train_r2,
                           'test_r2': test_r2,
                           'fitting_duration': end_time - start_time,
                           'series_type': series_type['Target']
                           }
        print(f"train MAPE: {train_MAPE}")
        print(f"test MAPE: {test_MAPE}")
        print(f"train MSE: {train_MSE}")
        print(f"test MSE: {test_MSE}")
        print(f"train MAE: {train_MAE}")
        print(f"test MAE: {test_MAE}")
        print(f"train R2: {train_r2}")
        print(f"test R2: {test_r2}")
        return training_result, grid_search_results
