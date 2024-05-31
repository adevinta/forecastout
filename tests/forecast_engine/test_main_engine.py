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
        25636100.0,
        29363566.0,
        28150300.0,
        24343184.0,
        24908472.0,
        10135740.0
    ]
    forecast_lower_list = [
        22820152.0,
        26547532.0,
        25334134.0,
        21526832.0,
        22091864.0,
         7318807.0
    ]
    forecast_upper_list = [
        28452048.0,
        32179600.0,
        30966468.0,
        27159540.0,
        27725078.0,
        12952672.0
    ]
    model_list = [
        "HW_A/HW_M",
        "HW_A/HW_M",
        "HW_A/HW_M",
        "HW_A/HW_M",
        "HW_A/HW_M",
        "HW_A/HW_M"
    ]

    return pd.DataFrame(
        {'forecast': np.float32(forecast_list),
         'forecast_lower': np.float32(forecast_lower_list),
         'forecast_upper': np.float32(forecast_upper_list),
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
        25120850.0,
        30190548.0,
        27727190.0,
        23190936.0,
        23293430.0,
        6018486.0,
        24452806.0,
        26819394.0,
        27106994.0,
        31620138.0,
        26501792.0,
        29798808.0,
        24258784.0,
        24427584.0,
        24832702.0,
        24984242.0,
        14573735.0,
        5697744.0
    ]
    forecast_lower_list = [
        25119470.0,
        30189138.0,
        27725568.0,
        23189284.0,
        23291702.0,
        6016737.5,
        18820912.0,
        26819394.0,
        21474926.0,
        31620138.0,
        20869460.0,
        29798808.0,
        18626078.0,
        24427584.0,
        19199490.0,
        24984240.0,
        8939870.0,
         5697744.0
    ]
    forecast_upper_list = [
        25122230.0,
        30191956.0,
        27728812.0,
        23192590.0,
        23295156.0,
        6020234.5,
        30084700.0,
        26819394.0,
        32739062.0,
        31620138.0,
        32134126.0,
        29798808.0,
        29891492.0,
        24427586.0,
        30465914.0,
        24984242.0,
        20207600.0,
        5697744.5
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
        {'date': pd.to_datetime(date_list),
         'forecast': np.float32(forecast_list),
         'forecast_lower': np.float32(forecast_lower_list),
         'forecast_upper': np.float32(forecast_upper_list),
         'model': model_list,
         }
    )


def models_ranked_output_df():

    model_list = [
        "HW_M",
        "HW_A",
        "autoarima"
    ]
    error_list = [
        2.123064e-08,
        2.285561e-01,
        2.421447e-01
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
