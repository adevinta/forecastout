# -- ML main packages
from abc import ABC, abstractmethod
import pandas as pd


class ForecastModel(ABC):

    def __init__(
            self,
            df_train_y: pd.DataFrame,
            dict_config: dict,
            series_train_dates: pd.Series
    ):
        self.df_train_y = df_train_y
        self.dict_config = dict_config
        self.series_train_dates = series_train_dates

    @abstractmethod
    def do_forecast(self, list_dates: list):
        pass

    @abstractmethod
    def get_feature_importance(self):
        pass
