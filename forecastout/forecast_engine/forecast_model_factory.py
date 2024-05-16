# Main forecasting forecast_models
# -- Python main packages
import pandas as pd

from forecastout.forecast_models.autoarima_model import AutoArimaModel
from forecastout.forecast_models.prophet_model import ProphetModel
from forecastout.forecast_models.holtwinters_model import HoltWintersModel
from forecastout.forecast_models.naive_seasonal import NaiveSeasonalModel


class ForecastModelFactory:
    @staticmethod
    def get_model(
            model,
            df_train_y: pd.DataFrame,
            dict_config: dict,
            series_train_dates: pd.Series
    ):
        if model == "autoarima":
            return AutoArimaModel(
                df_train_y=df_train_y,
                dict_config=dict_config
            )
        if model == "prophet":
            return ProphetModel(
                df_train_y=df_train_y,
                dict_config=dict_config,
                series_train_dates=series_train_dates
            )
        if model == "holtwinters":
            return HoltWintersModel(
                df_train_y=df_train_y,
                dict_config=dict_config,
                series_train_dates=series_train_dates
            )
        if model == "seasonalnaive":
            return NaiveSeasonalModel(
                df_train_y=df_train_y,
                dict_config=dict_config,
                series_train_dates=series_train_dates
            )
