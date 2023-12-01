from enum import Enum


class RegressionAlgorithmsEnum(Enum):
    AdaboostRegressor = 'Adaboost'
    SVR = 'SVR'
    RandomForestRegressor = 'RFR'
    LassoRegressor = 'Lasso'
    GaussianProcessRegressor = 'GPR'
    XGBoostRegressor = 'XGBoost'
    LightgbmRegressor = 'lightgbm'
    ElasticNetRegressor = 'ElasticNet'
    ExtraTreesRegressor = 'ExtraTrees'
