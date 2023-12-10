from enum import Enum


class ClassificationScoringMetricEnum(Enum):
    ACCURACY = 'accuracy'
    F1 = 'f1'
    ROC_AUC = 'roc_auc'
    PRECISION = 'precision'
    RECALL = 'recall'
