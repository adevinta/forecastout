# -- ML main packages
from abc import ABC, abstractmethod
import pandas as pd


class DisaggregationModel(ABC):

    def __init__(
            self,
            df_train_y: pd.DataFrame,
            df_train_x: pd.DataFrame,
            dict_config: dict
    ):
        self.df_train_y = df_train_y.copy()
        self.df_train_x = df_train_x.copy()
        self.dict_config = dict_config

    @abstractmethod
    def do_prediction(self, list_dates: list, df_test_x: pd.DataFrame):
        pass

    @abstractmethod
    def get_feature_importance(self):
        pass
