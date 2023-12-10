import pandas as pd
from sklearn.pipeline import Pipeline

from GizaAutoML.data_resampler.data_constructor import DataConstructor
from GizaAutoML.enums.aggregations_enums import AggregationsEnum
from GizaAutoML.enums.stages_enum import StagesEnum
from GizaAutoML.feature_engineering.data_preproccessing.estimators.series_type_estimator import SeriesTypeEstimator
from GizaAutoML.feature_engineering.data_preproccessing.transformers.MinMax_scaler_transformer import \
    MinMaxScalerTransformer
from GizaAutoML.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
from GizaAutoML.pipelines.preprocessing_pipeline import PreprocessingPipeline


class Utils:
    def __init__(self, series_col='Target', timestamp_col_name='Timestamp',
                 columns_to_exclude=None, is_forecast=True):
        if columns_to_exclude is None:
            columns_to_exclude = [timestamp_col_name]
        self.series_col = series_col
        self.timestamp_col_name = timestamp_col_name
        self.columns_to_exclude = columns_to_exclude
        self.features_extraction_stage_index = 0
        self.is_forecast = is_forecast

    @staticmethod
    def create_pipeline(processing_pipeline_stages):
        """Create sklearn pipeline"""
        pp_steps = [(stage.__class__.__name__ + f"_{idx}", stage) for idx, stage in
                    enumerate(processing_pipeline_stages)]
        return Pipeline(steps=pp_steps)

    def check_if_univariate(self, dataframe) -> bool:
        """
        Check if the DataFrame is univariate, meaning it contains only one variable other than the datetime column.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame to check.

        Returns:
        bool: True if the DataFrame is univariate, False otherwise.
        """
        df = dataframe.copy()
        columns = df.columns

        if self.timestamp_col_name in list(columns):
            df = df.set_index(self.timestamp_col_name)

        if len(columns) > 2:
            return False
        else:
            return True

    def prepare_univariate_data(self, dataframe):
        """ Rename dataframe columns to match the series and timestamp column names """
        df = dataframe.copy()
        if self.timestamp_col_name in df.columns:
            df.set_index(self.timestamp_col_name, inplace=True)
        df = df.rename(columns={df.columns[0]: self.series_col})
        df.reset_index(inplace=True)
        df[self.timestamp_col_name] = pd.to_datetime(df[self.timestamp_col_name])

        # sort values by date_col
        df.sort_values(by=[self.timestamp_col_name], inplace=True)
        print(df)
        return df

    def resample_data(self, dataframe):
        # todo: update for multivariate resampler
        df = dataframe.copy()
        if self.timestamp_col_name in df.columns:
            df.set_index(self.timestamp_col_name, inplace=True)
        # resample dataset
        constructor = DataConstructor(date_col=self.timestamp_col_name, target_col=self.series_col)
        resampled_df = constructor.resample(dataframe=df, agg_func=AggregationsEnum.AVG)
        return resampled_df

    @staticmethod
    def get_series_type(dataframe):
        # check series type
        st_estimator = SeriesTypeEstimator()
        series_type = st_estimator.series_type_estimator(dataframe)
        return series_type

    @staticmethod
    def _get_columns_to_encode(dataframe):
        """ get categorical columns in a dataframe """
        return dataframe.select_dtypes(include=['object']).columns.tolist()

    def _get_preprocessing_stages(self, df):
        preprocessing_stages = [StagesEnum.IMPUTER.name]

        if self._get_columns_to_encode(df):
            preprocessing_stages.append(StagesEnum.Encoder.name)

        return preprocessing_stages

    def _get_feature_extraction_stages(self):
        if self.is_forecast:
            feature_extraction_stages = [StagesEnum.TIME.name, StagesEnum.TREND.name,
                                         StagesEnum.SEASONALITY.name,
                                         StagesEnum.SHIFTER.name, StagesEnum.LAGGED.name]
            self.features_indexes = {'lagged_stage_index': -1, 'trend_stage_index': 1,
                                     'seasonality_stage_index': 2}
        else:
            feature_extraction_stages = [StagesEnum.TIME.name]
            self.features_indexes = None

        return feature_extraction_stages

    def get_processed_data(self, dataframe, series_type=None, preprocessing=True):
        """
        Apply preprocessing and feature extraction pipelines on the input dataframe.

        Args:
        - dataframe (pd.DataFrame): The dataframe to apply preprocessing and feature extraction on.

        Returns:
        - Tuple[pd.DataFrame, sklearn.pipeline.Pipeline]: A tuple containing the processed dataframe
          with preprocessing columns and the fitted preprocessing pipeline.

        """
        df = dataframe.copy()
        pipelines_list = []
        if not series_type:
            # check series type
            series_type = self.get_series_type(df)
        print(f"Series Type: {series_type}")

        # initialize preprocessing pipeline with imputer stage
        if preprocessing:
            stages = self._get_preprocessing_stages(dataframe)
            if stages:

                columns_to_impute = df.select_dtypes(include=['int64', 'float64']).columns

                processing_pipeline = PreprocessingPipeline(series_types=series_type,
                                                            stages=stages,
                                                            input_columns=[self.series_col]
                                                            if self.check_if_univariate(df)
                                                            else columns_to_impute)
                self.features_extraction_stage_index = 1
                pipelines_list.append(processing_pipeline)

        # initialize feature extraction pipeline
        feature_extraction_pipeline = FeatureExtractionPipeline(
            dataframe=df,
            stages=self._get_feature_extraction_stages(),
            series_types=series_type,
            is_forecast=self.is_forecast,
            target_col_name=self.series_col,
            timestamp_col=self.timestamp_col_name)
        pipelines_list.append(feature_extraction_pipeline)

        normalizer = MinMaxScalerTransformer(excluded_columns=self.columns_to_exclude)
        preprocessing_pipeline = self.create_pipeline(pipelines_list)

        print("......... Fit and Transform the Preprocessing pipeline on data .............")

        # fit on train data
        fitted_processing_pipeline = preprocessing_pipeline.fit(df)

        # predict on train data
        processed_data = fitted_processing_pipeline.transform(df)
        print(processed_data)
        return processed_data, fitted_processing_pipeline
