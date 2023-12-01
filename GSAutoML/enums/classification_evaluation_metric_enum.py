from enum import Enum
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from functools import partial  # So we can use the functions names and values in the evaluator class


class ClassificationEvaluationMetricEnum(Enum):
    # classification evaluation metrics
    # the enum members here are the metric functions, so we could use them directly
    ACCURACY = partial(accuracy_score)
    F1 = partial(f1_score, average='weighted')
    # ROC_AUC = partial(roc_auc_score, average='macro', multi_class= 'ovo')
    PRECISION = partial(precision_score, average='weighted')
    RECALL = partial(recall_score, average='weighted')
