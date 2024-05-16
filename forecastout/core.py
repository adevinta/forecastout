from forecastout.data_engine import DataHandler, TrainTestSplit
from forecastout.forecast_engine import (
    ModelTrainer,
    ModelPredictor,
    Backtester,
    ensemble_models,
    AnomalyFixer,
    create_df_ts_decomposition
)
from forecastout.disaggregation_features.add_features \
    import add_features
from forecastout.disaggregation_engine import (
    DisaggregationModelTrainer,
    DisaggregationModelPredictor,
    get_daily_shares,
    disaggregate_monthly_to_daily_forecast,
    remake_monthly_forecast_current_month
)
from forecastout.utils import update_config, InputChecker
from logging import Logger
import pandas as pd
import os.path as op
import yaml
import warnings
warnings.filterwarnings("ignore")


class ForecastOut:
    def __init__(
            self,
            df: pd.DataFrame,
            sum_aggregation: bool = True,
            horizon: int = 3,
            months_to_backtest: int = 3,
            models_to_use: list = None,
            average_top_models_number: int = 1
    ):
        # -- Initialize logs
        self.log = Logger(name="Initialization")
        # -- Initialize input
        self.df = df
        self.sum_aggregation = sum_aggregation
        self.horizon = horizon
        self.months_to_backtest = months_to_backtest
        self.models_to_use = models_to_use
        if self.models_to_use is None:
            self.models_to_use = ['autoarima', 'holtwinters', 'prophet']
        self.average_top_models_number = average_top_models_number
        InputChecker(
            df=self.df,
            sum_aggregation=self.sum_aggregation,
            horizon=self.horizon,
            months_to_backtest=self.months_to_backtest,
            models_to_use=self.models_to_use,
            average_top_models_number=self.average_top_models_number
        )
        # -- Initialize output
        self.df_monthly_forecast = pd.DataFrame()
        self.df_predictions_by_model = pd.DataFrame()
        self.df_ts_decomposition = pd.DataFrame()
        self.df_predictions_bt = pd.DataFrame()
        self.df_daily_shares = pd.DataFrame()
        self.df_daily_forecast = pd.DataFrame()
        # -- Read Config
        current_path = op.dirname(__file__)
        with open(op.join(current_path, "config.yaml"), 'rb') as fp:
            self.config = yaml.safe_load(fp)
        self.log.info(">> 0. Update Config")
        self.config = update_config(
            config=self.config,
            horizon=self.horizon,
            months_to_backtest=self.months_to_backtest,
            models_to_use=self.models_to_use,
            average_top_models_number=self.average_top_models_number
        )
        self.log.info(">> 1. Data Treatment")
        data_handler = DataHandler(
            df=self.df,
            horizon=self.config['horizon'],
            sum_aggregation=self.sum_aggregation
        )
        self.df_monthly = data_handler.df_monthly.copy()
        self.df_daily = data_handler.df_daily.copy()
        self.log.info(">> 2. Monthly Forecast")
        self.__make_monthly_forecast()
        self.log.info(">> 3. Daily Disaggregation")
        if data_handler.granularity_of_df == 'daily':
            self.__get_daily_disaggregation_shares()
            self.__make_daily_forecast()
            self.__remake_monthly_forecast()

    def __make_monthly_forecast(self):
        self.log.info(">> 2.1. Train/Test Split")
        df_train_test = TrainTestSplit.split_by_current_closed_month(
            df=self.df_monthly.copy(),
            current_month=(
                self.df_monthly.loc[
                    ~self.df_monthly['value'].isna(), 'date'
                ].max()
            ),
            granularity_month=True
        )
        self.log.info(">> 2.2. Train Models")
        model_trainer = ModelTrainer(
            model_names=self.config["models_to_use"],
            df_train=df_train_test[df_train_test['test'] == 0].copy(),
            config=self.config
        )
        df_models_trained = model_trainer.train()
        self.log.info(">> 2.3. Predict Models")
        model_predictor = ModelPredictor(
            df_models_trained=df_models_trained,
            df=df_train_test[df_train_test['test'] == 1].copy(),
            config=self.config,
        )
        self.df_predictions_by_model = model_predictor.predict()
        self.log.info(">> 2.4. Backtest Models")
        backtester = Backtester(
            df=self.df_monthly.copy(),
            model_names=self.config["models_to_use"],
            config=self.config
        )
        self.df_predictions_bt = backtester.return_predictions()
        df_models_ranked = backtester.return_models_ranked()
        self.log.info(">> 2.5. Ensemble Models")
        df_ensemble = ensemble_models(
            df_predictions=self.df_predictions_by_model.copy(),
            df_models_ranked=df_models_ranked.copy(),
            ensemble_method=self.config["ensemble"]["ensemble_method"],
            n_tops=self.config["ensemble"]["average_top_models"]["n_tops"]
        )
        self.log.info(">> 2.6. Correct anomaly forecast")
        anomaly_fixer = AnomalyFixer(
            df_forecast=df_ensemble.copy(),
            df_actual=df_train_test.copy(),
            config=self.config
        )
        self.df_monthly_forecast = anomaly_fixer.df_forecast.copy()
        self.log.info(">> 2.7. Time Series Decomposition")
        self.df_ts_decomposition = create_df_ts_decomposition(
            df=df_train_test.copy()
        )
        self.log.info(">> 2.X. End of forecast monthly")

    def __get_daily_disaggregation_shares(self):
        self.log.info(">> 3.1. Feature Creation")
        df = add_features(df=self.df_daily)
        self.log.info(">> 3.2. Train/Test Split")
        df_train_test = TrainTestSplit.split_by_current_closed_month(
            df=df.copy(),
            current_month=(
                self.df_monthly.loc[
                    ~self.df_monthly['value'].isna(), 'date'
                ].max()
            ),
            granularity_month=False
        )
        self.log.info(">> 3.3. Train disaggregation models")
        disaggregation_model_trainer = DisaggregationModelTrainer(
            df_train=df_train_test.loc[df_train_test['test'] == 0].copy(),
            config=self.config
        )
        disaggregation_model_trained = disaggregation_model_trainer.train(
            self.config["disaggregation_model_selected"]
        )
        self.log.info(">> 3.4. Predict disaggregation models")
        disaggregation_model_predictor = DisaggregationModelPredictor(
            model=disaggregation_model_trained,
            df=df_train_test.copy(),
            config=self.config
        )
        df_daily_prediction = disaggregation_model_predictor.predict()
        self.log.info(">> 3.5. Get daily shares")
        self.df_daily_shares = get_daily_shares(
            df_daily_prediction=df_daily_prediction.copy(),
            sum_aggregation=self.sum_aggregation
        )

    def __make_daily_forecast(self):
        df_daily_forecast = disaggregate_monthly_to_daily_forecast(
            df_daily_shares=self.df_daily_shares.copy(),
            df_monthly_forecast=self.df_monthly_forecast.copy()
        )
        self.df_daily_forecast = (
            df_daily_forecast.loc[
                df_daily_forecast["date"] > self.df["date"].max()
                ].copy()
        )

    def __remake_monthly_forecast(self):
        self.df_monthly_forecast = remake_monthly_forecast_current_month(
            df_daily_forecast=self.df_daily_forecast.copy(),
            df_actuals=self.df.copy()
        )
