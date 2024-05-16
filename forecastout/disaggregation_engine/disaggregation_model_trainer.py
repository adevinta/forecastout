import pandas as pd
from forecastout.disaggregation_engine.disaggregation_model_factory \
    import DisaggregationModelFactory


class DisaggregationModelTrainer:
    def __init__(
            self,
            df_train: pd.DataFrame,
            config: dict):
        self.df_train = df_train
        self.config = config

    @staticmethod
    def filter_config_by(config: dict, disaggregation_model: str):
        return config['disaggregation_models'][disaggregation_model]

    def train(self, disaggregation_model: str):
        model = DisaggregationModelFactory.get_model(
            disaggregation_model=disaggregation_model,
            df_train_y=self.df_train['value'],
            df_train_x=self.df_train[self.config["disaggregation_features"]],
            dict_config=DisaggregationModelTrainer.filter_config_by(
                self.config,
                disaggregation_model
            )
        )
        return model
