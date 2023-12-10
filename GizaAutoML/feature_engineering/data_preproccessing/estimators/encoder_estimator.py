from GizaAutoML.feature_engineering.common.estimator_interface import IEstimator
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from GizaAutoML.feature_engineering.data_preproccessing.transformers.encoder_transformer import \
    EncoderTransformer


class EncoderEstimator(IEstimator):
    """ ِA custom Estimator for applying categorical encoding"""

    def __init__(self):
        super(EncoderEstimator, self).__init__()
        self.columns_to_encode = []
        self.label_encoders = {}

    @staticmethod
    def _get_columns_to_encode(dataframe):
        """ get categorical columns in a dataframe """
        return dataframe.select_dtypes(include=['object']).columns.tolist()

    @staticmethod
    def accept(pipeline_visitor):
        return pipeline_visitor.get_encoder_instance()

    def fit(self,  X, y=None):
        """ Fit Label Encoder on the categorical columns in a dataframe """
        print(">>>>>>>>>> In Encoder estimator >>>>>>>>>>>>>>>>>")
        self.columns_to_encode = self._get_columns_to_encode(X)
        print(self.columns_to_encode)

        if self.columns_to_encode:
            for col in self.columns_to_encode:
                label_encoder = LabelEncoder()
                label_encoder.fit(X[col])
                self.label_encoders[col] = label_encoder
        return self

    def transform(self, X: pd.DataFrame):
        self.transformer = EncoderTransformer(self.columns_to_encode, self.label_encoders)
        return self.transformer.transform(X)
