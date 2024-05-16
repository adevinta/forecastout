import pandas as pd
import numpy as np
import holidays
from datetime import date


class FeatureCreator:

    def __init__(self, df: pd.DataFrame):
        # -- Main data
        self.df = df

    def get_lags(self, features: list, number_lags: int):
        for feature in features:
            for number_lag in range(1, (number_lags+1)):
                column_lag = feature + '_lag_' + str(number_lag)
                self.df[column_lag] = (
                    self.df[feature].shift(number_lag)
                )

    def get_moving_average(self, features: list, windows: list):
        for feature in features:
            for window in windows:
                column_moving_average = feature + '_ma_' + str(window)
                self.df[column_moving_average] = (
                    self.df[feature+'_lag_1']
                        .transform(lambda x: x.rolling(window, center=False)
                                   .mean()
                                   )
                )

    def get_day_of_week(self):
        self.df['day_of_week'] = self.df["date"].dt.dayofweek.astype(np.int64)

    def get_month(self):
        self.df['month'] = self.df["date"].dt.month.astype(np.int64)

    def get_year(self):
        self.df['year'] = self.df["date"].dt.year.astype(np.int64)

    def get_day(self):
        self.df['day'] = self.df["date"].dt.day.astype(np.int64)

    def get_holidays(self):
        # -- Add Spain Holidays
        spain_holidays = holidays.Spain()
        year_list = [i for i in range(2010, 2050)]
        # This is done because of a bug in the package holiday. Weird, yes.
        [date(year, 1, 2) in spain_holidays for year in year_list]
        df_holidays = pd.DataFrame.from_dict(
            spain_holidays,
            orient='index'
        ).reset_index()
        df_holidays["date"] = pd.to_datetime(df_holidays['index'])
        df_holidays['national_holiday'] = 1
        df_holidays.drop([0, 'index'], axis=1, inplace=True)
        # -- Merge holidays with main df
        self.df = self.df.merge(
            df_holidays,
            on="date",
            how='left'
        )
        self.df['national_holiday'] = self.df['national_holiday'].fillna(0)
        # -- Add weekends
        self.df['national_holiday_or_weekend'] = (
            np.where(
                (self.df['national_holiday'] == 1)
                | (self.df['day_of_week'].isin([5, 6])
                   ),
                1,
                0
            )
        )
        # -- Add long weekends
        self.df['puente_1'] = self.df['national_holiday_or_weekend'].shift(1)
        self.df['puente_minus1'] = (
            self.df['national_holiday_or_weekend'].shift(-1)
        )
        self.df['long_weekends'] = (
            np.where((self.df['national_holiday_or_weekend'] == 0)
                     & (self.df['puente_1'] == 1)
                     & (self.df['puente_minus1'] == 1),
                     1,
                     self.df['national_holiday_or_weekend'])
        )
        self.df.drop(["puente_1", "puente_minus1"], axis=1, inplace=True)
