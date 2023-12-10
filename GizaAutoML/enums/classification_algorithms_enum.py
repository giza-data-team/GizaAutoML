from enum import Enum


class ClassificationAlgorithmsEnum(Enum):
    AdaboostClassifier = 'Adaboost'
    SVC = 'SVC'
    RandomForestClassifier = 'RFC'
    LassoClassifier = 'Lasso'  # L1
    GaussianProcessClassifier = 'GPC'
    XGBoostClassifier = 'XGBoost'
    LightgbmClassifier = 'lightgbm'
    ElasticNetClassifier = 'ElasticNet'  # L1+L2
