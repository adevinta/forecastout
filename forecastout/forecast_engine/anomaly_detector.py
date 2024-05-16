import pandas as pd


class AnomalyDetector:
    """
    This class determines if there is an anomaly in the forecast
    """

    def __init__(self,
                 df_forecast: pd.DataFrame,
                 df_actual: pd.DataFrame,
                 horizon: int
                 ):
        # -- Main data
        self.df_forecast = df_forecast
        self.df_actual = df_actual
        self.horizon = horizon
        self.columns_id = []
        # -- Ensure order
        self.df_forecast.sort_values(["date"], inplace=True)
        self.df_actual.sort_values(["date"], inplace=True)
        # -- Initialize anomaly 0
        self.anomaly = 0
        # -- Ensure order
        self.__get_explosive_trends()
        self.__get_non_seasonals()
        self.__get_negatives()

    def __get_explosive_trends(self):
        """
        Check trend
        """
        df_forecast = self.df_forecast.copy()
        df_actual = self.df_actual.copy()
        # -- Check all forecast period.
        forecast_allperiod = df_forecast['forecast'].sum()
        # -- Check all actual periods.
        df_actual_allperiod = df_actual.tail(self.horizon * 2)
        actual_period1 = \
            df_actual_allperiod.head(self.horizon)['value'].sum()
        actual_period2 = \
            df_actual_allperiod.tail(self.horizon)['value'].sum()
        # -- Forecast is  < abs( than period1 - period2 dev)*2
        dev_12 = abs(actual_period2 - actual_period1) / actual_period1
        dev_forecast2 = (
                abs(forecast_allperiod - actual_period2) / actual_period2
        )
        if dev_forecast2 > dev_12 * 2:
            self.anomaly = 1

    def __get_non_seasonals(self):
        """
        Check seasonality
        """
        df_forecast = self.df_forecast.copy()
        df_actual = self.df_actual.copy()
        # -- forecast manipulation
        df_forecast = df_forecast.head(
            self.horizon if self.horizon <= 12 else 12
        )
        df_forecast['month'] = df_forecast["date"].dt.month
        df_forecast = df_forecast[["month", "forecast"]].copy()
        # -- actual manipulation
        df_actual = (
            df_actual.tail(self.horizon if self.horizon <= 12 else 12)
        )
        df_actual['month'] = df_actual["date"].dt.month
        df_actual = df_actual[["month", "value"]].copy()
        df_allperiod = df_forecast.merge(
            df_actual,
            on=["month"],
            how='left'
        )
        df_allperiod_corr = df_allperiod[['forecast', 'value']].corr()
        if df_allperiod_corr.iloc[0, 1] < 0.5:
            self.anomaly = 1

    def __get_negatives(self):
        """
        Check negative forecast
        """
        df_forecast = self.df_forecast.copy()
        if len(df_forecast.loc[df_forecast['forecast'] <= 0]) > 0:
            self.anomaly = 1
