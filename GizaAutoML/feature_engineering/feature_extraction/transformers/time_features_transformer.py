import pandas as pd
from GizaAutoML.feature_engineering.common.transformer_interface import ITransformer


class TimeFeaturesTransformer(ITransformer):
    def __init__(self, timestamp_col_name=None):
        super().__init__()
        self.set_timestamp_col_name(timestamp_col_name)
        self.target_col_name = "Target"

    def set_timestamp_col_name(self, timestamp_col):
        self.timestamp_col_name  = timestamp_col
        return self

    def get_timestamp_col_name(self):
        return self.timestamp_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df_pandas: pd.DataFrame):
        print(">>>>>>>>>> In time feature transformer >>>>>>>>>>>>>>>>>")
        if self.get_timestamp_col_name() not in df_pandas.columns:
            df_pandas.reset_index(inplace=True)
        hod = self.__get_hour_of_day(df_pandas).to_list()
        dow = self.__get_day_of_week(df_pandas).to_list()
        moy = self.__get_month_of_year(df_pandas).to_list()

        idx = df_pandas[self.get_timestamp_col_name()].to_list()
        data = {'HourOfDay': hod, 'DayOfWeek': dow, 'MonthOfYear': moy, self.get_timestamp_col_name(): idx}

        # -->x df_added = ps.DataFrame(data, index=df.index.values)  CAUSE an error,
        # because data frame has no index!! Is it supposed to be has an index? if yes should be reset first

        df_added = pd.DataFrame(data).set_index(self.get_timestamp_col_name())
        df_pandas.set_index(self.get_timestamp_col_name(), inplace=True, drop=True)
        # df_pandas = df_pandas.reset_index(drop=True)
        all_df = pd.concat([df_pandas, df_added], axis=1)
        # shift target column one step backward to predict the forecasted data at t+1
        # all_df[self.target_col_name] = all_df[self.target_col_name].shift(-1)
        # all_df = all_df.fillna(method='ffill')
        all_df.reset_index(inplace=True)
        #all_df = all_df.dropna()
        return all_df

    def __get_day_of_week(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the day of the week of a given date as an integer.
        And return a new ps.Series with DayOfWeek extracted.
        :param df: Pandas dataframe
        :return ps.Series
        """
        dow_ser = df[self.get_timestamp_col_name()].dt.dayofweek
        return dow_ser

    def __get_month_of_year(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the month of the year of a given date as an integer.
        And return a new ps.Series with MonthOfYear extracted.
        :param df: Pandas dataframe
        :return ps.Series
        """
        moy_ser = df[self.get_timestamp_col_name()].dt.month
        return moy_ser

    def __get_hour_of_day(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the hour of the day of a given date as an integer.
        And return a new ps.Series with HourOfDay extracted.
        :param df: Pandas dataframe
        :return ps.Series
        """
        hod_ser = df[self.get_timestamp_col_name()].dt.hour
        return hod_ser
