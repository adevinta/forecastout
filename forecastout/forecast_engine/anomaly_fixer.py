import pandas as pd
from forecastout.forecast_engine.anomaly_detector import AnomalyDetector
from forecastout.forecast_engine.model_trainer import ModelTrainer
from forecastout.forecast_engine.model_predictor import ModelPredictor


class AnomalyFixer:
    def __init__(
            self,
            df_forecast: pd.DataFrame,
            df_actual: pd.DataFrame,
            config: dict
    ):
        self.df_forecast = df_forecast
        self.df_actual = df_actual
        self.config = config
        # -- Anomalies
        anomaly_detector = AnomalyDetector(
            df_forecast=df_forecast.copy(),
            df_actual=df_actual[df_actual['test'] == 0].copy(),
            horizon=self.config['horizon']
        )
        self.anomaly = anomaly_detector.anomaly
        self.__apply_recursive_naiveseasonal()

    def __apply_recursive_naiveseasonal(self):
        i = 0
        while i < 100 and self.anomaly > 0:
            # -- Forecast with models_to_fix
            model_trainer = ModelTrainer(
                model_names=self.config["models_to_fix"],
                df_train=self.df_actual[self.df_actual['test'] == 0].copy(),
                config=self.config
            )
            df_models_trained = model_trainer.train()
            model_predictor = ModelPredictor(
                df_models_trained=df_models_trained,
                df=self.df_actual[self.df_actual['test'] == 1].copy(),
                config=self.config,
            )
            df_predictions = model_predictor.predict()
            # -- Correct forecast
            if i > 0:
                df_predictions['model'] = ''
            model_sep = '' if i > 0 else '/'
            df_forecast_corrected = pd.concat(
                [self.df_forecast, df_predictions]
            )
            self.df_forecast = df_forecast_corrected.groupby(["date"]).agg({
                'forecast': 'mean',
                'forecast_lower': 'mean',
                'forecast_upper': 'mean',
                'model': lambda x: model_sep.join(x)}).reset_index()
            anomaly_detector = AnomalyDetector(
                df_forecast=self.df_forecast.copy(),
                df_actual=(
                    self.df_actual[self.df_actual['test'] == 0].copy()
                ),
                horizon=self.config['horizon']
            )
            # -- Check repetition
            self.anomaly = anomaly_detector.anomaly
            i += 1
        
