from sklearn.pipeline import Pipeline

from GizaAutoML.data_modeling.estimators.regression_estimator import RegressionEstimator
from GizaAutoML.data_modeling.estimators.statistical_estimator import StatisticalEstimator
from GizaAutoML.data_modeling.regressors import Regressors
from GizaAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum
from GizaAutoML.pipelines.pipeline_interface import MachineLearningPipeline


class RegressorPipeline(MachineLearningPipeline):
    """
    Pipeline for regression tasks using various regression algorithms.

    Methods:
        create_estimator: Create and return the regression estimator object.
        _get_estimator: Get the regression estimator object.
        _get_estimator_params_grid: Get regression estimator parameters for grid search.
        _get_stages: Get pipeline stages specific to regression.
        fit(X, y, **kwargs): Fits the regression pipeline to the training data.
        transform(X): Transforms the input data using the trained pipeline.
    """
    def create_estimator(self):
        """ Create and return the classifier object specific to the task (classification). """
        return Regressors(self.estimator_name, self.label_col_name, self.time_stamp_col_name, self.seasonality_mode)

    def _get_estimator(self):
        """ get regressor object """
        return self.estimator.get_regressor()

    def _get_estimator_params_gird(self):
        """ get regressor params for grid search """
        return self.estimator.get_regressor_params()

    def _get_stages(self):
        """ get pipeline stages """
        if self.estimator.regressor_name in RegressionAlgorithmsEnum.__members__.values():
            regressor = RegressionEstimator(label_col=self.label_col_name,
                                            prediction_col=self.prediction_col_name,
                                            timestamp_col_name=self.time_stamp_col_name,
                                            scoring_metric=self.scoring_metric,
                                            estimator=self._get_estimator(),
                                            params=self._get_estimator_params_gird(),
                                            exclude_cols=self.exclude_cols,
                                            hyperparameters=self.hyperparameters,
                                            algorithm_name=self.estimator_name)
        else:
            regressor = StatisticalEstimator(label_col=self.label_col_name,
                                             prediction_col=self.prediction_col_name,
                                             scoring_metric=self.scoring_metric,
                                             estimator=self._get_estimator(),
                                             params=self._get_estimator_params_gird(),
                                             )
        return [('regressor', regressor)]

    def fit(self, X, y=None, **kwargs):
        pipeline = Pipeline(steps=self.steps)
        self.pipeline = pipeline.fit(X=X, y=y)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return X

    def predict(self,X):
        X = self.pipeline.predict(X)
        return X