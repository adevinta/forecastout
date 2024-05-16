from forecastout.forecast_models.abstract_model import ForecastModel
from statsforecast.models import HoltWinters
from statsforecast import StatsForecast
import pandas as pd
import numpy as np


class HoltWintersModel(ForecastModel):
    def __init__(self,
                 series_train_dates: pd.Series,
                 *args,
                 **kwargs
                 ):
        """
        This implementation returns the corresponding `ETS` model with additive
        (A) or multiplicative (M) errors (so either 'AAA' or 'MAM').
        """
        super(HoltWintersModel, self).__init__(*args, **kwargs)
        # -- Constructor
        # -- Initialize Holtwinters Exponential Smoothing with additive and
        # multiplicative methods
        self.holtwinters_model = StatsForecast(models=[
            HoltWinters(
                season_length=self.dict_config['freq'],
                error_type='A',
                alias='HW_A'),
            HoltWinters(
                season_length=self.dict_config['freq'],
                error_type='M',
                alias='HW_M')],
            freq='MS',
            n_jobs=-1
        )
        # -- Transform data
        df_train_y_holtwinters = pd.concat(
            [series_train_dates, self.df_train_y],
            axis=1
        )
        df_train_y_holtwinters.columns = ['ds', 'y']
        df_train_y_holtwinters['ds'] = (
            pd.to_datetime(df_train_y_holtwinters['ds'])
        )
        df_train_y_holtwinters['ds'] = (
            df_train_y_holtwinters['ds'].dt.tz_localize(None)
        )
        df_train_y_holtwinters['unique_id'] = 'id'
        # -- Fit forecast_models
        self.holtwinters_model.fit(df_train_y_holtwinters)

    def do_forecast(self, list_dates: list) -> pd.DataFrame:
        # -- Predict
        holtwinters_prediction = self.holtwinters_model.predict(
            h=len(list_dates),
            level=[self.dict_config['alpha_intervals']]
        ).reset_index()
        # -- Transform data for correct output
        value_vars = [
            col for col in holtwinters_prediction.columns
            if 'HW' in col
        ]
        holtwinters_prediction_melted = pd.melt(
            holtwinters_prediction,
            id_vars=["unique_id", "ds"],
            value_vars=value_vars,
            var_name='model_var',
            value_name='value'
        )
        holtwinters_prediction_melted['model'] = (
            holtwinters_prediction_melted['model_var']
            .str.split('-', expand=True)[0]
        )
        holtwinters_prediction_melted['model_metric'] = (
            np.where((
                         holtwinters_prediction_melted['model_var']
                         .str
                         .contains('lo')
                     ) == True,
                     'forecast_lower',
                     np.where((
                                  holtwinters_prediction_melted['model_var']
                                  .str
                                  .contains('hi')
                              ) == True,
                              'forecast_upper',
                              'forecast')
                     )
        )
        holtwinters_prediction = holtwinters_prediction_melted.pivot(
            index=['ds', 'model'],
            columns="model_metric",
            values='value'
        ).reset_index()
        holtwinters_prediction.columns.name = None
        holtwinters_prediction.rename(columns={'ds': 'date'}, inplace=True)
        # -- Ensure date
        holtwinters_prediction['date'] = (
            pd.to_datetime(holtwinters_prediction['date'])
        )
        return holtwinters_prediction

    def get_feature_importance(self):
        return None
