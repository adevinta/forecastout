import pandas as pd
from typing import List
from forecastout.forecast_engine.forecast_model_factory \
    import ForecastModelFactory


class ModelTrainer:
    def __init__(
            self,
            model_names: List[str],
            df_train: pd.DataFrame,
            config: dict
    ):
        self.model_names = model_names
        self.df_train = df_train
        self.config = config

    @staticmethod
    def filter_config_by(config: dict, model) -> dict:
        return config['models'][model]

    def train(self) -> pd.DataFrame:
        list_output = []
        # --  Training:
        for model_name in self.model_names:
            dict_output_id = {}
            model = ForecastModelFactory.get_model(
                model=model_name,
                df_train_y=self.df_train['value'],
                dict_config=ModelTrainer.filter_config_by(
                    self.config,
                    model_name
                ),
                series_train_dates=self.df_train['date']
            )
            dict_output_id['model'] = model
            list_output.append(dict_output_id)
        return pd.DataFrame(list_output)
