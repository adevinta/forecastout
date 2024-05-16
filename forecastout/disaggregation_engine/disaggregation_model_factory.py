import pandas as pd
from forecastout.disaggregation_models.random_forest_model \
    import RandomForestModel


class DisaggregationModelFactory:
    @staticmethod
    def get_model(
            disaggregation_model: str,
            df_train_y: pd.DataFrame,
            df_train_x: pd.DataFrame,
            dict_config: dict
    ):
        if disaggregation_model == "random_forest":
            return RandomForestModel(
                df_train_y=df_train_y,
                df_train_x=df_train_x,
                dict_config=dict_config
            )
