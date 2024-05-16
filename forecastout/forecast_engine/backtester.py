# Main forecasting forecast_models

# -- Python main packages
from forecastout.data_engine.train_test_split import TrainTestSplit
from forecastout.forecast_engine.model_trainer import ModelTrainer
from forecastout.forecast_engine.model_predictor import ModelPredictor
from typing import List
import pandas as pd


class Backtester:
    def __init__(self, df: pd.DataFrame, model_names: List[str], config: dict):
        self.df = df
        self.model_names = model_names
        self.config = config
        self.df_predictions = pd.DataFrame()
        self.df_models_ranked = pd.DataFrame()
        # -- start process
        self.__make_forecast_for_validation()
        self.__rank_models()

    def __make_forecast_for_validation(self):
        # --  BT. Train/Test Split
        df = TrainTestSplit.split_for_ts_backtest(
            df=self.df,
            current_month=self.df.loc[~self.df['value'].isna(), 'date'].max(),
            months_to_backtest=self.config["months_to_backtest"])
        # --  BT. Train Models
        model_trainer = ModelTrainer(
            model_names=self.config["models_to_use"],
            df_train=df[df['test'] == 0].copy(),
            config=self.config
        )
        df_models_trained = model_trainer.train()
        # -- BT. Predict Models
        model_predictor = ModelPredictor(
            df_models_trained=df_models_trained,
            df=df[df['test'] == 1].copy(),
            config=self.config,
        )
        df_predictions = model_predictor.predict()
        # -- BT. Merge actual and predictions
        df_predictions = df_predictions.merge(
            df.loc[df['test'] == 1, ["date", "value"]].copy(),
            on="date",
            how='left'
        )
        self.df_predictions = (
            df_predictions[["date", "model", "forecast", "value"]]
            .copy()
        )

    def __rank_models(self):
        # -- BT. Rank Models
        df_predictions = self.df_predictions.copy()
        df_predictions['abs_error'] = (
                abs((df_predictions['value'] - df_predictions['forecast']))
                / df_predictions['value']
        )
        df_ranking = (
            df_predictions
            .groupby(["model"])['abs_error']
            .mean()
            .reset_index()
        )
        df_ranking.rename(
            columns={'abs_error': 'mean_abs_error'},
            inplace=True
        )
        df_ranking.sort_values(["mean_abs_error"], inplace=True)
        df_ranking['ranking'] = 1
        df_ranking['ranking'] = df_ranking['ranking'].cumsum()
        self.df_models_ranked = df_ranking.copy()

    def return_predictions(self) -> pd.DataFrame:
        return self.df_predictions

    def return_models_ranked(self) -> pd.DataFrame:
        return self.df_models_ranked
