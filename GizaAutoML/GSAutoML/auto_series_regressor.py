from GizaAutoML.controller.common.knowledge_base_collector import KnowledgeBaseCollector
from GizaAutoML.controller.common.utils import Utils
from datetime import datetime

from GizaAutoML.controller.meta_model_controllers.current_regression import CurrentRegression
from GizaAutoML.enums.regression_grid_search_scoring_enum import RegressionScoringMetricEnum
import pandas as pd
from GizaAutoML.enums.dataset_types_enums import DatasetTypeEnum


class AutoSeriesRegressor:
    def __init__(self, raw_dataframe,
                 sorting_metric=RegressionScoringMetricEnum.MAE.name,
                 time_budget=10,
                 save_results=False,
                 random_seed=1,
                 processed_dataframe=pd.DataFrame(),
                 target_col="Target",
                 dataset_name=None,
                 dataset_instance=None):
        self.dataframe = raw_dataframe
        self.processed_dataframe = processed_dataframe
        self.sorting_metric = sorting_metric
        self.time_budget = time_budget
        self.target_col = target_col
        self.utils = Utils(series_col=self.target_col, is_forecast=False)
        self.exceptions = []
        self.save_results_flag = save_results
        self.random_seed = random_seed
        self.kb_collector = KnowledgeBaseCollector()
        self.dataset_name = dataset_name if dataset_name else str(datetime.now().strftime("%H%M%S%f")[:-3])
        dataset = {'name': self.dataset_name, 'type': DatasetTypeEnum.Uni.value if self.utils.check_if_univariate(
            self.dataframe) else DatasetTypeEnum.Multi.value}
        self.kb_collector.add_dataset(dataset)
        self.dataset_instance = dataset_instance if dataset_instance else self.kb_collector.get_dataset(
            self.dataset_name)

    def fit(self):
        self.meta_model = CurrentRegression(raw_dataframe=self.dataframe,
                                            processed_dataframe=self.processed_dataframe,
                                            sorting_metric=self.sorting_metric,
                                            time_budget=self.time_budget,
                                            random_seed=self.random_seed,
                                            target_col=self.target_col,
                                            dataset_name=self.dataset_name,
                                            dataset_instance=self.dataset_instance,
                                            save_results=self.save_results_flag,
                                            utils=self.utils)
        self.meta_model.run()
        self.pipeline = self.meta_model.parent_pipeline
        return self.pipeline

    def transform(self, X):
        return self.pipeline.transform(X)
