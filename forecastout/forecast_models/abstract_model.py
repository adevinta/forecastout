# -- ML main packages
from abc import ABC, abstractmethod
import pandas as pd

class ForecastModel(ABC):

    def __init__(
            self,
            df_train_y:pd.DataFrame,
            dict_config:dict
    ):
        self.df_train_y = df_train_y
        self.dict_config = dict_config

    @abstractmethod
    def do_forecast(self, list_dates: list):
        pass

    @abstractmethod
    def get_feature_importance(self):
        pass
