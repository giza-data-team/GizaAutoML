import pandas as pd

from GSAutoML.controller.common.knowledge_base_client import KnowledgeBaseClient
from GSAutoML.enums.regression_algorithms_enum import RegressionAlgorithmsEnum


class KnowledgeBaseCollector:

    def __init__(self):
        self.kb_client = KnowledgeBaseClient()

    def get_meta_features(self):
        meta_features = self.kb_client.get_uni_variate_meta_features()
        meta_features_df = pd.DataFrame(meta_features)
        print(meta_features_df)
        df = pd.json_normalize(meta_features_df['dataset'])
        meta_features_df['dataset_name'] = df['name']
        meta_features_df.drop(['id', 'dataset'], axis=1, inplace=True)
        print(meta_features_df.info())
        return meta_features_df

    def add_uni_variate_meta_features(self, meta_features):
        response = self.kb_client.add_uni_variate_meta_features(data=meta_features)
        print(response.json())

    def add_multi_variate_meta_features(self, meta_features):
        response = self.kb_client.add_multi_variate_meta_features(data=meta_features)
        print(response.json())

    def get_dataset(self, dataset_name):
        response = self.kb_client.get_datasets(dataset_name)
        return response[0]

    def add_dataset(self, dataset_info):
        response = self.kb_client.add_dataset(data=dataset_info)
        print(response.json())
        return response.json()

    def get_regression_algorithms_performance(self):
        alg_performance = self.kb_client.get_algorithms_performance()
        alg_performance_df = pd.DataFrame(alg_performance)
        print(alg_performance_df)
        df = pd.json_normalize(alg_performance_df['dataset'])
        alg_performance_df['dataset_name'] = df['name']
        alg_performance_df.drop(['id', 'dataset'], axis=1, inplace=True)
        print(alg_performance_df)
        alg_performance_df.drop_duplicates(inplace=True)
        # drop lasso trials
        alg_performance_df = alg_performance_df[alg_performance_df['algorithm']!= RegressionAlgorithmsEnum.LassoRegressor.name]
        # todo: select datasets with 3 or more algorithms

        return alg_performance_df

    def add_regression_algorithms_performance(self, data):
        response = self.kb_client.add_algorithms_performance(data)
        print(response.json())
