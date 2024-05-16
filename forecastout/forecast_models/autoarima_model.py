from forecastout.forecast_models.abstract_model import ForecastModel
from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np


class AutoArimaModel(ForecastModel):
    """
    Auto arima model
    """
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(AutoArimaModel, self).__init__(*args, **kwargs)
        self.arima_model = (
            auto_arima(self.df_train_y,
                       seasonal=self.dict_config["seasonality"],
                       m=self.dict_config["freq"]
                       )
        )

    def do_forecast(self, list_dates: list) -> pd.DataFrame:
        arima_prediction, arima_conf_int = self.arima_model.predict(
            len(list_dates),
            return_conf_int=True,
            alpha=self.dict_config["alpha_intervals"]
        )
        arima_prediction = pd.concat(
            [
                pd.DataFrame(
                    np.array(arima_prediction.reset_index()[0]),
                    columns=['forecast']),
                pd.DataFrame(
                    arima_conf_int,
                    columns=['forecast_lower', 'forecast_upper'])
            ],
            axis=1)
        arima_prediction['model'] = 'autoarima'
        arima_prediction['date'] = list_dates
        # -- Ensure date
        arima_prediction['date'] = pd.to_datetime(arima_prediction['date'])
        return arima_prediction

    def get_feature_importance(self):
        return None
