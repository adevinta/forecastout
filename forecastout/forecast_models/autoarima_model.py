from forecastout.forecast_models.abstract_model import ForecastModel
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
import pandas as pd


class AutoArimaModel(ForecastModel):
    """
    Auto arima model
    """
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(AutoArimaModel, self).__init__(*args, **kwargs)
        # -- Constructor
        # -- Initialize autoarima
        self.autoarima_model = StatsForecast(models=[
            AutoARIMA(
                season_length=self.dict_config['freq'],
                seasonal=self.dict_config['seasonality'])],
            freq='MS',
            n_jobs=-1
        )
        # -- Transform data
        df_train_y_autoarima = (
            pd.concat([self.series_train_dates, self.df_train_y], axis=1)
        )
        df_train_y_autoarima.columns = ['ds', 'y']
        df_train_y_autoarima['ds'] = (
            pd.to_datetime(df_train_y_autoarima['ds'])
        )
        df_train_y_autoarima['ds'] = (
            df_train_y_autoarima['ds'].dt.tz_localize(None)
        )
        df_train_y_autoarima['unique_id'] = 'id'
        # -- Fit forecast_models
        self.autoarima_model.fit(df_train_y_autoarima)

    def do_forecast(self, list_dates: list) -> pd.DataFrame:
        # -- Predict
        autoarima_prediction = self.autoarima_model.predict(
            h=len(list_dates),
            level=[self.dict_config['alpha_intervals']]
        ).reset_index()
        # -- Transform data for correct output
        autoarima_prediction = (
            autoarima_prediction.drop(['unique_id'], axis=1)
        )
        autoarima_prediction.columns = [
            'date', 'forecast', 'forecast_lower', 'forecast_upper'
        ]
        autoarima_prediction['model'] = 'autoarima'
        autoarima_prediction['date'] = list_dates
        # -- Ensure date
        autoarima_prediction['date'] = (
            pd.to_datetime(autoarima_prediction['date'])
        )
        return autoarima_prediction

    def get_feature_importance(self):
        return None
