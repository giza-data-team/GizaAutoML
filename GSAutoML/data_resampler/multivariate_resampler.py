import pandas as pd

from GSAutoML.data_resampler.resampler_interface import ResamplerInterface
from GSAutoML.enums.aggregations_enums import AggregationsEnum
from GSAutoML.enums.intervals_enum import IntervalEnum


class MultivariateResampler(ResamplerInterface):
    def __init__(self, date_col="Timestamp", target_col="Target"):
        super().__init__(date_col, target_col)

    def resample_data(self, dataframe: pd.DataFrame, interval: int, time_unit: IntervalEnum,
                      aggregate_func: AggregationsEnum = AggregationsEnum.AVG,
                      multivariate_aggregate_func: dict = None,
                      origin: str = 'start'):

        # check if the index is of any the supported types
        dataframe = self._set_date_index(dataframe)

        # Check if no aggregations specified, default aggregation is used
        if not multivariate_aggregate_func:
            multivariate_aggregate_func = self._set_default_aggregations(dataframe, self.target_col)

        # Prepare aggregations dictionary
        aggregations = {key: func.value for key, func in multivariate_aggregate_func.items()}
        aggregations[self.target_col] = aggregate_func.value

        # Check for custom aggregations
        fields_with_custom_agg = []
        category_fields_added = []
        for key, value in aggregations.items():
            # Apply the MODE aggregations
            if value == AggregationsEnum.MODE.value:
                aggregations[key] = lambda key: key.mode().iloc[0] if not key.mode().empty else None

            # Apply the CATEGORIESCOUNT aggregations
            if value == AggregationsEnum.CATEGORIESCOUNT.value:
                dataframe, categories = self._apply_one_hot_encoding(dataframe, key)
                fields_with_custom_agg.append(key)
                category_fields_added.extend(categories)

        # Combine all aggregations
        aggregations = {key: value for key, value in aggregations.items() if key not in fields_with_custom_agg}
        category_fields_agg = {field: AggregationsEnum.SUM.value for field in category_fields_added}
        aggregations.update(category_fields_agg)

        resampling_df = dataframe.resample(str(interval) + time_unit.value, origin=origin).agg(aggregations)

        return resampling_df

    @staticmethod
    def _apply_one_hot_encoding(dataframe: pd.DataFrame, field_name: str):
        categories_df = pd.get_dummies(dataframe[field_name], prefix=field_name)
        categories = categories_df.columns
        dataframe = pd.concat([dataframe, categories_df], axis=1)
        dataframe.drop(columns=[field_name], inplace=True)
        return dataframe, categories

    @staticmethod
    def _set_default_aggregations(df: pd.DataFrame, target_col) -> dict:
        df = df.drop(target_col, axis=1)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

        numerical_agg = {key: AggregationsEnum.AVG for key in numerical_cols}
        categorical_agg = {key: AggregationsEnum.MODE for key in categorical_cols}

        return {**numerical_agg, **categorical_agg}
