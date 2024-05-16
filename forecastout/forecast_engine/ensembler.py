import pandas as pd


class Ensembler:
    """
    This class ensembles forecast_models' forecasts to provide a unique
    forecast
    """

    @staticmethod
    def average_top_models(
            df_predictions: pd.DataFrame,
            df_models_ranked: pd.DataFrame,
            n_tops: int
    ) -> pd.DataFrame:
        """
        Average top forecast_models
        """
        # -- Get top forecast_models for each id
        df_models_ranked = (
            df_models_ranked
            .loc[df_models_ranked["ranking"] <= n_tops]
        )
        # -- Filter unwanted forecast_models
        df_predictions = df_predictions.merge(
            df_models_ranked,
            on=["model"],
            how='left'
        )
        df_predictions = (
            df_predictions.loc[~df_predictions['ranking'].isna()]
            .copy()
        )
        # -- Average
        df_predictions.sort_values(["model"], inplace=True)
        df_predictions = df_predictions.groupby(['date']).agg(
            {"forecast": "mean",
             "forecast_lower": "mean",
             "forecast_upper": "mean",
             "model": '/'.join}
        ).reset_index()
        forecast_columns = ["forecast", "forecast_lower", "forecast_upper"]
        df_predictions = df_predictions[forecast_columns + ["date", "model"]]
        return df_predictions

    @staticmethod
    def get_best_models(
            df_predictions: pd.DataFrame,
            df_models_ranked: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get best model
        """
        # -- Get top forecast_models for each id
        df_models_ranked = (
            df_models_ranked
            .loc[df_models_ranked["ranking"] == 1]
        )
        # -- Filter unwanted forecast_models
        df_predictions = df_predictions.merge(
            df_models_ranked,
            on=["model"],
            how='left'
        )
        df_predictions = (
            df_predictions
            .loc[~df_predictions['ranking'].isna()].copy()
        )
        forecast_columns = ["forecast", "forecast_lower", "forecast_upper"]
        df_predictions = df_predictions[forecast_columns + ["date", "model"]]
        return df_predictions
