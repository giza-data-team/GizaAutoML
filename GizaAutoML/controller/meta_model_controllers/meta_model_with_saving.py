import pandas as pd

from GizaAutoML.controller.meta_model_controllers.meta_model_template import MetaModelTemplate
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
from GizaAutoML.split_data.add_lags_test_data import LaggedDataPreprocessor
from GizaAutoML.split_data.split_data import TimeSeriesSplitter


class MetaModelWithSaving(MetaModelTemplate):

    def __int__(self, raw_dataframe, processed_dataframe=pd.DataFrame(),
                sorting_metric=RegressionScoringMetricEnum.MAE.name,
                save_results=True,
                time_budget=10,
                random_seed=1,
                target_col="Target",
                dataset_name=None,
                dataset_instance=None,
                utils=None):
        super().__init__(raw_dataframe, processed_dataframe,
                         sorting_metric,
                         time_budget, save_results, random_seed, target_col,
                         dataset_name,
                         dataset_instance,
                         utils)

    def get_split_data(self, dataframe):
        train_data, test_data = TimeSeriesSplitter(value_column=self.target_col,
                                                   timestamp_column=self.timestamp_col_name).split_data(data=dataframe)
        return train_data, test_data

    def get_test_results(self, train_data, test_data, last_significant_lags_index, parent_pipeline,
                         algorithms_recommender):
        test_lagged_processor = LaggedDataPreprocessor(train_data=train_data, test_data=test_data,
                                                       num_lags=last_significant_lags_index)
        test_data = test_lagged_processor.preprocess_data()
        test_data_with_predictions = parent_pipeline.transform(test_data)
        print("......... Evaluate the pipeline on test data .............")
        test_result = algorithms_recommender.get_scores(test_data_with_predictions)
        return test_result

    def get_training_results(self, best_hyperparameters, training_results, test_result, fitting_duration,
                             extracted_features_info, best_algorithm_name, series_type, training_time):
        training_result = {'dataset': self.dataset_instance['id'],
                           'train_MAPE': training_results['MAPE'],
                           'test_MAPE': test_result['MAPE'],
                           'train_MSE': training_results['MSE'],
                           'test_MSE': test_result['MSE'],
                           'train_MAE': training_results['MAE'],
                           'test_MAE': test_result['MAE'],
                           'train_r2_score': training_results['R2'],
                           'test_r2_score': test_result['R2'],
                           'fitting_duration': fitting_duration,
                           'training_duration': training_time/60,
                           'series_type': series_type[self.target_col],
                           'trend_type': extracted_features_info['trend_type'],
                           'seasonality_components_no': extracted_features_info['seasonality_components_no'],
                           'lags_no': extracted_features_info['last_significant_lags_index'],
                           'algorithm': best_algorithm_name,
                           'hyperparameters':str(best_hyperparameters),
                           'GS_scoring_metric': self.sorting_metric
                           }
        self.kb_collector.add_regression_algorithms_performance(training_result)
        return training_result

