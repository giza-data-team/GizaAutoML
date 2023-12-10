from sklearn.pipeline import Pipeline

from GizaAutoML.data_modeling.estimators.regressor_estimator_factory import RegressorEstimatorFactory
from GizaAutoML.enums.automl_engines_enum import AutomlEnginesEnum

class AutomlPipeline(Pipeline):
    def __init__(self, label_col, prediction_col, exclude_cols, scoring_metric, random_seed,
                 engine: AutomlEnginesEnum, time_budget: int, time_stamp_col_name=None,
                 seasonality_mode=None):
        self.label_col_name = label_col
        self.prediction_col_name = prediction_col
        self.scoring_metric = scoring_metric
        self.time_stamp_col_name = time_stamp_col_name
        self.seasonality_mode = seasonality_mode
        self.exclude_cols = exclude_cols
        self.time_budget = time_budget
        self.random_seed = random_seed
        self.engine = engine
        self.steps = self._get_stages()

    def _get_stages(self):
        """ Get pipeline stages """
        regressor = RegressorEstimatorFactory().create_regressor_estimator(engine=self.engine,
                                                                           label_col=self.label_col_name,
                                                                           prediction_col=self.prediction_col_name,
                                                                           exclude_cols=self.exclude_cols,
                                                                           scoring_metric=self.scoring_metric,
                                                                           time_budget=self.time_budget,
                                                                           random_seed=self.random_seed)

        return [('regressor', regressor)]

    def fit(self, X, y=None, **kwargs):
        pipeline = Pipeline(steps=self.steps)
        self.pipeline = pipeline.fit(X=X, y=y)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return X
