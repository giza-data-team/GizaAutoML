from enum import Enum
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor


class AlgorithmsMappingEnum(Enum):
    AdaboostRegressor = AdaBoostRegressor
    SVR = SVR
    RandomForestRegressor = RandomForestRegressor
    LassoRegressor = linear_model.LassoCV
    GaussianProcessRegressor = GaussianProcessRegressor
    XGBoostRegressor = xgb.XGBRegressor
    LightgbmRegressor = lgb.LGBMRegressor
    ElasticNetRegressor = linear_model.ElasticNetCV
    ExtraTreesRegressor = ExtraTreesRegressor
