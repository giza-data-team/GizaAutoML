from enum import Enum
from GSAutoML.feature_engineering.feature_extraction.estimators.lagged_features_estimator import \
    LaggedFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.seasonality_features_estimator import \
    SeasonalityFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.stationary_features_estimator import \
    StationaryFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.time_features_estimator import \
    TimeFeaturesEstimator
from GSAutoML.feature_engineering.feature_extraction.estimators.trend_feature_estimator import \
    TrendFeatureEstimator
from GSAutoML.feature_engineering.data_preproccessing.estimators.prophet_imputer_estimator import \
    ProphetImputerEstimator
from GSAutoML.feature_engineering.data_preproccessing.estimators.constant_columns_removal_estimator import \
    ConstantColumnsRemovalEstimator
from GSAutoML.feature_engineering.data_preproccessing.estimators.encoder_estimator import \
    EncoderEstimator
from GSAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer import \
    MinMaxScalerTransformer
from GSAutoML.feature_engineering.common.shift_target import TargetShifter
from GSAutoML.feature_engineering.feature_extraction.transformers.column_replacer import ColumnReplacer


class StagesEnum(Enum):
    SEASONALITY = SeasonalityFeaturesEstimator
    LAGGED = LaggedFeaturesEstimator
    STATIONARY = StationaryFeaturesEstimator
    TIME = TimeFeaturesEstimator
    TREND = TrendFeatureEstimator
    IMPUTER = ProphetImputerEstimator
    ConstantColumns = ConstantColumnsRemovalEstimator
    Encoder = EncoderEstimator
    NORMALIZER = MinMaxScalerTransformer
    SHIFTER = TargetShifter
    COLUMN_REPLACER = ColumnReplacer
