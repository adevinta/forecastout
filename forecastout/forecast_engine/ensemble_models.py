from forecastout.forecast_engine.ensembler import Ensembler
import pandas as pd


def ensemble_models(
        df_predictions: pd.DataFrame,
        df_models_ranked: pd.DataFrame,
        ensemble_method: str,
        n_tops: int
) -> pd.DataFrame:
    """Select the specified algorithm"""
    if ensemble_method == "get_best_models":
        df_ensemble = Ensembler.get_best_models(
            df_predictions=df_predictions,
            df_models_ranked=df_models_ranked
        )
        return df_ensemble
    if ensemble_method == "average_top_models":
        df_ensemble = Ensembler.average_top_models(
            df_predictions=df_predictions.copy(),
            df_models_ranked=df_models_ranked.copy(),
            n_tops=n_tops
        )
        return df_ensemble
