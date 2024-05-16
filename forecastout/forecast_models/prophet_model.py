from forecastout.forecast_models.abstract_model import ForecastModel
from prophet import Prophet
import pandas as pd


class ProphetModel(ForecastModel):
    def __init__(self,
                 series_train_dates: pd.Series,
                 *args,
                 **kwargs
                 ):
        """
        This implementation a prophet model
        """
        super(ProphetModel, self).__init__(*args, **kwargs)
        # -- Prophet
        # -- Constructor
        df_train_y_prophet = pd.concat(
            [series_train_dates, self.df_train_y],
            axis=1)
        df_train_y_prophet.columns = ['ds', 'y']
        df_train_y_prophet['ds'] = pd.to_datetime(df_train_y_prophet['ds'])
        df_train_y_prophet['ds'] = (
            df_train_y_prophet['ds']
            .dt
            .tz_localize(None)
        )
        prophet = Prophet()
        self.prophet_model = prophet.fit(df_train_y_prophet)

    def do_forecast(self, list_dates: list) -> pd.DataFrame:
        future = self.prophet_model.make_future_dataframe(
            periods=len(list_dates),
            freq="MS",
            include_history=False
        )
        prophet_prediction = (
            self.prophet_model
            .predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        )
        prophet_prediction.columns = ['date', 'forecast', 'forecast_lower',
                                      'forecast_upper']
        prophet_prediction['model'] = 'prophet'
        prophet_prediction['date'] = list_dates
        # -- Ensure date
        prophet_prediction['date'] = pd.to_datetime(prophet_prediction['date'])
        return prophet_prediction

    def get_feature_importance(self):
        return None
