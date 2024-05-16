import numpy as np
import pandas as pd
from forecastout.forecast_engine.model_trainer import ModelTrainer
from forecastout.forecast_engine.model_predictor import ModelPredictor
from forecastout.forecast_engine.backtester import Backtester
from forecastout.forecast_engine.ensemble_models import ensemble_models

MODELS_TO_USE = ['autoarima', 'holtwinters']


def test_monthly_main_engine():
    """
    This function tests the following classes:
    - Ensembler
    - ForecastModelFactory
    - ModelPredictor
    - ModelTrainer
    - Backtester
    """
    # -- Get inputs
    df_input = input_df()
    config = config_input_dict()
    # -- Main development
    model_trainer = ModelTrainer(
        model_names=MODELS_TO_USE,
        df_train=df_input[df_input['test'] == 0].copy(),
        config=config
    )
    df_models_trained = model_trainer.train()
    model_predictor = ModelPredictor(
        df_models_trained=df_models_trained,
        df=df_input[df_input['test'] == 1].copy(),
        config=config,
    )
    df_predictions = model_predictor.predict()
    backtester = Backtester(
        df=df_input.copy(),
        model_names=MODELS_TO_USE,
        config=config
    )
    df_models_ranked = backtester.return_models_ranked()
    df_ensemble = ensemble_models(
        df_predictions=df_predictions.copy(),
        df_models_ranked=df_models_ranked.copy(),
        ensemble_method=config["ensemble"]["ensemble_method"],
        n_tops=config["ensemble"]["average_top_models"]["n_tops"]
    )
    # -- Tests
    pd.testing.assert_frame_equal(
        df_predictions.reset_index(drop=True),
        predictions_output_df()
    )
    pd.testing.assert_frame_equal(
        df_models_ranked.reset_index(drop=True),
        models_ranked_output_df().reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        df_ensemble,
        ensemble_output_df()
    )


def input_df():
    date_list = [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
        "2021-05-01",
        "2021-06-01",
        "2021-07-01",
        "2021-08-01",
        "2021-09-01",
        "2021-10-01",
        "2021-11-01",
        "2021-12-01",
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-01",
        "2022-06-01",
        "2022-07-01",
        "2022-08-01",
        "2022-09-01",
        "2022-10-01",
        "2022-11-01",
        "2022-12-01",
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
        "2023-08-01",
        "2023-09-01",
        "2023-10-01",
        "2023-11-01",
        "2023-12-01",
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01"
    ]
    value_list = [
        5830302.908,
        7400457.853,
        7449702.162,
        6480787.829,
        6995587.495,
        1675807.097,
        8372644.865,
        11324372.84,
        1799333.476,
        12702195.21,
        4509435.271,
        13316086.92,
        12826666.4,
        15473684.6,
        14899404.32,
        12463053.52,
        12991805.35,
        3016452.775,
        14652128.51,
        19318047.79,
        2998889.127,
        20724634.3,
        7215096.433,
        20925279.44,
        19823029.89,
        23546911.35,
        22349106.49,
        18445319.21,
        18988023.2,
        4357098.453,
        20931612.16,
        27311722.74,
        4198444.778,
        28747073.38,
        9920757.596,
        28534471.96,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan
    ]
    test_list = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'test': test_list}
    )


def ensemble_output_df():
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01"
    ]
    forecast_list = [
        2.649917e+07,
        3.113824e+07,
        2.931963e+07,
        2.431952e+07,
        2.482144e+07,
        6.939827e+06
    ]
    forecast_lower_list = [
        2.425845e+07,
        2.888590e+07,
        2.690031e+07,
        2.190020e+07,
        2.240212e+07,
        4.520505e+06
    ]
    forecast_upper_list = [
        2.873989e+07,
        3.339058e+07,
        3.173895e+07,
        2.673885e+07,
        2.724076e+07,
        9.359149e+06
    ]
    model_list = [
        "HW_M/autoarima",
        "HW_M/autoarima",
        "HW_M/autoarima",
        "HW_M/autoarima",
        "HW_M/autoarima",
        "HW_M/autoarima"
    ]

    return pd.DataFrame(
        {'forecast': forecast_list,
         'forecast_lower': forecast_lower_list,
         'forecast_upper': forecast_upper_list,
         'date': pd.to_datetime(date_list),
         'model': model_list
         }
    )


def predictions_output_df():
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01",
        "2024-01-01",
        "2024-01-01",
        "2024-02-01",
        "2024-02-01",
        "2024-03-01",
        "2024-03-01",
        "2024-04-01",
        "2024-04-01",
        "2024-05-01",
        "2024-05-01",
        "2024-06-01",
        "2024-06-01"
    ]
    forecast_list = [
        2.617895e+07,
        3.065635e+07,
        2.884046e+07,
        2.421146e+07,
        2.465864e+07,
        8.181910e+06,
        2.445281e+07,
        2.681939e+07,
        2.710699e+07,
        3.162014e+07,
        2.650179e+07,
        2.979881e+07,
        2.425878e+07,
        2.442758e+07,
        2.483270e+07,
        2.498424e+07,
        1.457374e+07,
        5.697744e+06
    ]
    forecast_lower_list = [
        2.169750e+07,
        2.615167e+07,
        2.400181e+07,
        1.937282e+07,
        1.982000e+07,
        3.343266e+06,
        1.882091e+07,
        2.681939e+07,
        2.147493e+07,
        3.162014e+07,
        2.086946e+07,
        2.979881e+07,
        1.862608e+07,
        2.442758e+07,
        1.919949e+07,
        2.498424e+07,
        8.939870e+06,
        5.697744e+06
    ]
    forecast_upper_list = [
        3.066039e+07,
        3.516102e+07,
        3.367910e+07,
        2.905011e+07,
        2.949728e+07,
        1.302055e+07,
        3.008470e+07,
        2.681939e+07,
        3.273906e+07,
        3.162014e+07,
        3.213413e+07,
        2.979881e+07,
        2.989149e+07,
        2.442759e+07,
        3.046591e+07,
        2.498424e+07,
        2.020760e+07,
        5.697744e+06
    ]
    model_list = [
        "autoarima",
        "autoarima",
        "autoarima",
        "autoarima",
        "autoarima",
        "autoarima",
        "HW_A",
        "HW_M",
        "HW_A",
        "HW_M",
        "HW_A",
        "HW_M",
        "HW_A",
        "HW_M",
        "HW_A",
        "HW_M",
        "HW_A",
        "HW_M"
    ]

    return pd.DataFrame(
        {'forecast': forecast_list,
         'forecast_lower': forecast_lower_list,
         'forecast_upper': forecast_upper_list,
         'model': model_list,
         'date': pd.to_datetime(date_list),
         }
    )


def models_ranked_output_df():

    model_list = [
        "HW_M",
        "autoarima",
        "HW_A"
    ]
    error_list = [
        2.123064e-08,
        7.825683e-02,
        2.285561e-01
    ]
    ranking_list = [
        1,
        2,
        3
    ]
    return pd.DataFrame(
        {'model': model_list,
         'mean_abs_error': error_list,
         'ranking': ranking_list
         }
    )


def config_input_dict():
    config = {
        "horizon": 15,
        "months_to_backtest": 3,
        "models_to_use": ['autoarima', 'holtwinters'],
        "models_to_fix": ['seasonalnaive'],
        "models": {
            "autoarima": {
                "seasonality": "True",
                "freq": 12,
                "alpha_intervals": 0.05},
            "holtwinters": {
                "freq": 12,
                "alpha_intervals": 95}
        },
        "ensemble": {
            "ensemble_method": "average_top_models",
            "average_top_models": {"n_tops": 2}
        }
    }
    return config
