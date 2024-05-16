# -- Packages
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
from functools import reduce


class DataHandler:
    """
    This class treats the necessary data to perform the Forecast
    """

    def __init__(self, df: pd.DataFrame, horizon: int, sum_aggregation: bool):
        # -- Main data and parameters
        self.df = df
        self.horizon = horizon
        self.sum_aggregation = sum_aggregation
        self.granularity_of_df = 'monthly'
        # -- Methods to compute granularity
        self.__ensure_date()
        self.__ensure_value()
        self.__detect_data_granularity()
        df_daily, df_monthly = self.__create_granular_dfs()
        # -- Clean df_monthly
        df_monthly = self.__expand_dates(df_monthly, freq='MS')
        df_monthly = self.__nan_to_zero(df_monthly)
        df_monthly = self.__get_order(df_monthly)
        self.df_monthly = df_monthly.copy()
        if self.granularity_of_df == 'daily':
            # -- Clean df_daily
            df_daily = self.__expand_dates(df_daily, freq='D')
            df_daily = self.__nan_to_zero(df_daily)
            df_daily = self.__get_order(df_daily)
            self.df_daily = df_daily.copy()
        else:
            self.df_daily = pd.DataFrame()

    def __ensure_date(self):
        self.df['date'] = pd.to_datetime(self.df['date'])

    def __ensure_value(self):
        self.df['value'] = self.df['value'].astype(float)

    def __detect_data_granularity(self):
        if any(
                self.df.groupby(
                    self.df['date'].astype(str).str[0:7]
                )['date'].count() > 1):
            self.granularity_of_df = "daily"

    def __create_granular_dfs(self) -> (pd.DataFrame, pd.DataFrame):
        df = self.df.copy()
        if self.granularity_of_df == 'monthly':
            df_daily = None
            df_monthly = df.copy()
        else:
            # -- compute relevant dates to clean dataframes
            last_data_date = df['date'].max()
            last_data_day = last_data_date.day
            max_month_day = reduce(
                lambda x, y: y[1],
                [calendar.monthrange(
                    last_data_date.year,
                    last_data_date.month)],
                None
            )
            # -- clean daily dataframe
            if last_data_day < max_month_day:
                first_month_day = pd.to_datetime(
                    str(last_data_date)[0:7] + '-01'
                )
                df_daily = df[df["date"] < first_month_day].copy()
            else:
                df_daily = df.copy()
            # -- clean monthly dataframe
            if self.sum_aggregation:
                df_monthly = (
                    (
                        df_daily
                        .groupby(df['date'].astype(str).str[0:7])['value']
                        .sum()
                        .reset_index()
                    )
                )
            else:
                df_monthly = (
                    (
                        df_daily
                        .groupby(df['date'].astype(str).str[0:7])['value']
                        .mean()
                        .reset_index()
                    )
                )
            df_monthly['date'] = pd.to_datetime(df_monthly['date'] + '-01')
            df_monthly['value'] = df_monthly['value'].astype(float)
        return df_daily, df_monthly

    def __expand_dates(
            self,
            df_granular: pd.DataFrame,
            freq: str
    ) -> pd.DataFrame():
        # -- Get first day of month and max date to disaggregate
        min_date = df_granular["date"].max().replace(day=1)
        if self.granularity_of_df == 'monthly':
            max_date = min_date + relativedelta(months=self.horizon)
        else:
            max_date = (
                    min_date
                    + relativedelta(months=self.horizon+1)
                    + relativedelta(days=-1)
            )
        # -- dates df
        dates = pd.date_range(
            start=df_granular["date"].min(),
            end=max_date,
            freq=freq
        ).tolist()
        df_dates = pd.DataFrame(
            range(len(dates)),
            dates
        ).reset_index()
        df_dates.columns = ["date", 'index']
        # -- Add values
        df_granular = df_dates.merge(
            df_granular,
            on='date',
            how='left'
        )
        df_granular = df_granular.drop('index', axis=1)
        return df_granular

    @staticmethod
    def __get_order(df_granular: pd.DataFrame) -> pd.DataFrame:
        df_granular.sort_values("date", inplace=True)
        return df_granular

    @staticmethod
    def __nan_to_zero(df_granular: pd.DataFrame) -> pd.DataFrame:
        max_actual = pd.to_datetime(df_granular.dropna()['date'].max())
        df_granular.loc[
            (df_granular["value"].isnull())
            & (df_granular["date"] <= max_actual),
            "value"
        ] = 0
        return df_granular
