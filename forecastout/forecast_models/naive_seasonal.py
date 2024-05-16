from forecastout.forecast_models.abstract_model import ForecastModel
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
import pandas as pd


class NaiveSeasonalModel(ForecastModel):
    def __init__(self,
                 series_train_dates: pd.Series,
                 *args,
                 **kwargs
                 ):
        """
        This implementation a naive seasonal forecast
        """
        super(NaiveSeasonalModel, self).__init__(*args, **kwargs)
        # -- Constructor
        # -- Initialize Holtwinters Exponential Smoothing with additive
        # and multiplicative methods
        self.seasonalnaive_model = StatsForecast(models=[
            SeasonalNaive(season_length=self.dict_config['freq'])],
            freq='MS',
            n_jobs=-1
        )
        # -- Transform data
        df_train_y_seasonalnaive = (
            pd.concat([series_train_dates, self.df_train_y], axis=1)
        )
        df_train_y_seasonalnaive.columns = ['ds', 'y']
        df_train_y_seasonalnaive['ds'] = (
            pd.to_datetime(df_train_y_seasonalnaive['ds'])
        )
        df_train_y_seasonalnaive['ds'] = (
            df_train_y_seasonalnaive['ds'].dt.tz_localize(None)
        )
        df_train_y_seasonalnaive['unique_id'] = 'id'
        # -- Fit forecast_models
        self.seasonalnaive_model.fit(df_train_y_seasonalnaive)

    def do_forecast(self, list_dates: list) -> pd.DataFrame:
        # -- Predict
        seasonalnaive_prediction = self.seasonalnaive_model.predict(
            h=len(list_dates),
            level=[self.dict_config['alpha_intervals']]
        ).reset_index()
        # -- Transform data for correct output
        seasonalnaive_prediction = (
            seasonalnaive_prediction.drop(['unique_id'], axis=1)
        )
        seasonalnaive_prediction.columns = [
            'date', 'forecast', 'forecast_lower', 'forecast_upper'
        ]
        seasonalnaive_prediction['model'] = 'seasonalnaive'
        seasonalnaive_prediction['date'] = list_dates
        # -- Ensure date
        seasonalnaive_prediction['date'] = (
            pd.to_datetime(seasonalnaive_prediction['date'])
        )
        return seasonalnaive_prediction

    def get_feature_importance(self):
        return None
