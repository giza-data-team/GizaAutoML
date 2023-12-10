import pickle
from typing import List
from pathlib import Path
import pandas as pd
from GizaAutoML.enums.regression_evaluation_metric_enum import RegressionEvaluationMetricEnum
from GizaAutoML.controller.common.utils import Utils
from GizaAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GizaAutoML.enums.ML_tasks_enum import MLTasksEnum
from GizaAutoML.pipelines.ML_pipeline_factory import MLFactory
from GizaAutoML.data_modeling.evaluator import Evaluator


class MetaModelAlgorithmsRecommender:
    """
       class for recommending regression algorithms based on meta features and performance data.

       This class provides functionality to recommend the top-performing regression algorithms
       for a given dataset based on its meta features and past performance data.

       Args:
           series (DataFrame): The dataset or time series data as a DataFrame.
           series_meta_features (dict): Meta features extracted from the dataset.
           sorting_metric (str): The metric used for sorting and ranking the algorithms.

       Attributes:
           series (DataFrame): The dataset or time series data.
           series_meta_features (dict): Meta features extracted from the dataset.
           sorting_metric (str): The metric used for sorting and ranking algorithms.
           utils (Utils): An instance of the utility class for common operations.
           target_col_name (str): The name of the target column in the dataset.
           prediction_col_name (str): The name of the column for storing algorithm predictions.
           time_stamp_col_name (str): The name of the timestamp column if applicable.
       """

    def __init__(self, series, series_meta_features,
                 sorting_metric, prediction_col_name='prediction',
                 time_stamp_col_name='Timestamp', target_col_name='Target', is_forecast=True):
        self.series = series
        self.series_meta_features = series_meta_features

        # TODO: Enhance this!!!!!
        path = Path(__file__).parent / "meta_model.pkl"
        with path.open('rb') as f:
            self.meta_model = pickle.load(f)
        path_2 = Path(__file__).parent / "univariate_meta_features_le.pkl"
        with path_2.open('rb') as f:
            self.label_encoders_dict = pickle.load(f)
        self.sorting_metric = sorting_metric
        self.is_forecast = is_forecast
        self.target_col_name = target_col_name
        self.prediction_col_name = prediction_col_name
        self.time_stamp_col_name = time_stamp_col_name
        self.utils = Utils(series_col=self.target_col_name, is_forecast=self.is_forecast)

    def get_recommended_algorithms(self):
        """
        Retrieves the best algorithms to start with for the warm start of the SMAC model.

        This method takes the meta-features of the new dataset as inference data to the RFC MetaModel and returns the
        predicted algorithm for the warm start

        Returns:
            a list of Algorithm names
        """
        new_meta_features_df = pd.DataFrame([self.series_meta_features])
        new_meta_features_df.set_index('dataset_name', inplace=True)

        for column_name, encoder in self.label_encoders_dict.items():
            new_meta_features_df[column_name] = encoder.transform(new_meta_features_df[column_name].astype("category"))

        return self.predict_best_3_algorithms(new_meta_features_df)

    def predict_best_3_algorithms(self, df):
        """
        This method takes predicted probabilities of the different classes and returns the
        3 algorithms with the highest probability.

        Args:
            df: a pandas DataFrame with algorithms are the columns names and the predicted probability
            of each class is the value
        Returns:
            list of algorithms name
        """
        predicted_probabilities = \
            pd.Series(self.meta_model.predict_proba(df[self.meta_model.feature_names_in_])[0], index=self.meta_model.classes_)
        print(self.series.shape)

        # exclude Gaussian

        if RegressionAlgorithmsEnum.GaussianProcessRegressor.name in predicted_probabilities.index:
                    predicted_probabilities = predicted_probabilities.\
                        drop(RegressionAlgorithmsEnum.GaussianProcessRegressor.name)

        # exclude svr if dataset size is larger than 10000 instance
        if self.series.shape[0] > 5000:
            algorithms_to_exclude = [RegressionAlgorithmsEnum.SVR.name]
            for algorithm in algorithms_to_exclude:
                if algorithm in predicted_probabilities.index:
                    predicted_probabilities = predicted_probabilities.drop(algorithms_to_exclude)

        print(predicted_probabilities)
        recommended_algorithms = predicted_probabilities.nlargest(2).index.tolist()
        recommended_algorithms.append(RegressionAlgorithmsEnum.ExtraTreesRegressor.name)
        return recommended_algorithms

    def get_algorithms_configuration(self, algorithm_names: List[str]):
        """
        Retrieves the configurations of top-performing algorithms trained on the closest dataset.

        This method utilizes performance data and selects the algorithms with the best performance
        on the specified closest dataset. It returns a DataFrame containing information about the
        selected algorithms and their hyperparameters.

        Args:
            algorithm_names: list of recommended algorithms

        Returns:
            algorithms_config (DataFrame): A DataFrame containing algorithm names and their corresponding
                                           hyperparameters for the top-performing algorithms on the specified
                                           closest dataset. GaussianProcessRegressor is excluded if the dataset
                                           size is larger than 10,000 instances for memory issues.
        """
        configs = {'AdaboostRegressor': {'learning_rate': 0.1, 'loss': 'exponential', 'n_estimators': 50},
                   'ElasticNetRegressor': {'l1_ratio': 0.5},
                   'GaussianProcessRegressor': {'alpha': 0.01},
                   'LassoRegressor': {'alpha': 9.999999999999999e-06},
                   'LightgbmRegressor': {'bagging_freq': 1, 'boosting_type': 'gbdt', 'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'min_gain_to_split': 0.1, 'n_estimators': 50, 'num_leaves': 100, 'reg_lambda': 0.1, 'reg_alpha':0.1},
                   'RandomForestRegressor': {'max_depth': 5, 'n_estimators': 50},
                   'SVR': {'C': 5, 'epsilon': 0.01, 'kernel': 'linear'},
                   'XGBoostRegressor': {'colsample_bytree': 0.8, 'gamma': 0.9, 'learning_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'reg_lambda': 5, 'subsample':0.5},
                   'ExtraTreesRegressor': {'n_estimators': 512, 'max_features': 1, 'min_samples_split': 19, 'min_samples_leaf': 4, 'bootstrap': True, 'criterion': 'friedman_mse', 'warm_start': True}
                   }

        hyperparameters = []
        for algorithm in algorithm_names:
            hyperparameters.append(configs[algorithm])

        algorithms_config = pd.DataFrame({'algorithm': algorithm_names, 'hyperparameters': hyperparameters})

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
            prediction_col=self.prediction_col_name,
            scoring_metric=self.sorting_metric,
            estimator=RegressionAlgorithmsEnum[algorithm_name],
            exclude_cols=[],
            time_stamp_col_name=self.time_stamp_col_name,
            seasonality_mode=series_type,
            hyperparameters=hyperparameters
        )

        fitted_modeling_pipeline = modeling_pipeline.fit(X=df)
        train_data_with_predictions = fitted_modeling_pipeline.transform(df)
        training_results = self.get_scores(train_data_with_predictions)

        return training_results, fitted_modeling_pipeline,train_data_with_predictions
