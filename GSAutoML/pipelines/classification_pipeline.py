from GSAutoML.pipelines.pipeline_interface import MachineLearningPipeline
from sklearn.pipeline import Pipeline
from GSAutoML.data_modeling.estimators.classification_estimator import ClassificationEstimator
from GSAutoML.data_modeling.classifiers import Classifiers


class ClassificationPipeline(MachineLearningPipeline):
    """
    A machine learning pipeline designed for classification tasks.

    Methods:
        create_estimator():
            Create and return the classifier object specific to the task (classification).

        _get_estimator():
            Get the classifier object.

        _get_estimator_params_grid():
            Get classifier hyperparameters for grid search.

        _get_stages():
            Get the stages of the classification pipeline.

        fit(X, y=None, **kwargs):
            Fit the classification pipeline to the training data.

        transform(X):
            Transform the input data using the trained pipeline.

        """
    def create_estimator(self):
        """ Create and return the classifier object specific to the task (classification). """
        return Classifiers(self.estimator_name, self.label_col_name, self.time_stamp_col_name, self.seasonality_mode)

    def _get_estimator(self):
        """ Get classifier object """
        return self.estimator.get_classifier()

    def _get_estimator_params_gird(self):
        """ get classifier params for grid search """
        return self.estimator.get_classifier_params()

    def _get_stages(self):
        """ get pipeline stages """
        classifier = ClassificationEstimator(label_col=self.label_col_name,
                                             prediction_col=self.prediction_col_name,
                                             time_stamp_col_name=self.time_stamp_col_name,
                                             scoring_metric=self.scoring_metric,
                                             estimator=self._get_estimator(),
                                             params=self._get_estimator_params_gird(),
                                             exclude_cols=self.exclude_cols,
                                             hyperparameters=self.hyperparameters,
                                             algorithm_name=self.estimator_name)
        return [('classifier', classifier)]

    def fit(self, X, y=None, **kwargs):
        pipeline = Pipeline(steps=self.steps)
        self.pipeline = pipeline.fit(X=X, y=y)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return X
