import pandas as pd


class ModelPredictor:
    def __init__(
            self,
            df_models_trained:
            pd.DataFrame,
            df: pd.DataFrame,
            config: dict
    ):
        self.df_models_trained = df_models_trained
        self.df = df
        self.config = config

    def predict(self) -> pd.DataFrame:
        list_output = []
        for model in self.df_models_trained['model'].unique():
            # -- Get test
            dates_test = self.df['date'].tolist()
            # -- Do forecast
            df_forecast_i = model.do_forecast(
                list_dates=dates_test
            )
            # -- Append
            list_output.append(df_forecast_i)
        # -- Convert to df
        df_output = pd.concat(list_output)
        # -- Ensure positivity
        df_num = df_output._get_numeric_data()
        df_num[df_num < 0] = 0
        return df_output
