from GizaAutoML.enums.automl_engines_enum import AutomlEnginesEnum
import importlib


class RegressorEstimatorFactory:
    """
    Factory class for creating regressor estimators based on the selected AutoML engine.

    Attributes:
        ENGINE_CLASS_MAPPING (dict): A mapping of AutoML engine to the corresponding estimator class name.

    Methods:
        create_regressor_estimator(engine, label_col, prediction_col, exclude_cols, scoring_metric, time_budget, random_seed):
            Create a regressor estimator based on the selected AutoML engine.

    """
    ENGINE_CLASS_MAPPING = {
        AutomlEnginesEnum.tpot: "TPOTRegressorEstimator",
        AutomlEnginesEnum.auto_sklearn: "AutoSKLearnRegressorEstimator",
    }

    def create_regressor_estimator(self, engine, label_col, prediction_col, exclude_cols, scoring_metric, time_budget,
                                   random_seed):
        """
        Create a regressor estimator based on the selected AutoML engine.

        Args:
            engine (AutomlEnginesEnum): The selected AutoML engine.
            label_col (str): The name of the label column.
            prediction_col (str): The name of the prediction column.
            exclude_cols (list): A list of columns to exclude from modeling.
            scoring_metric (str): The scoring metric for model evaluation.
            time_budget (int): The time budget for autoML model training.
            random_seed (int): The random seed for reproducibility.

        Returns:
            A regressor estimator instance based on the selected AutoML engine.

        Raises:
            ValueError: If an invalid engine type is provided. Supported engines are 'tpot' and 'auto_sklearn'.
        """
        if engine == AutomlEnginesEnum.tpot:
            estimator_module = importlib.import_module("GSAutoML.data_modeling.estimators.tpot_estimator")
        elif engine == AutomlEnginesEnum.auto_sklearn:
            estimator_module = importlib.import_module("GSAutoML.data_modeling.estimators.auto_sklearn_estimator")
        else:
            raise ValueError("Invalid engine type. Supported engines are 'tpot' and 'auto_sklearn'.")
        estimator_class_name = self.ENGINE_CLASS_MAPPING[engine]
        regressor_class = getattr(estimator_module, estimator_class_name)
        return regressor_class(label_col=label_col, prediction_col=prediction_col, exclude_cols=exclude_cols,
                               scoring_metric=scoring_metric, time_budget=time_budget, random_seed=random_seed)
