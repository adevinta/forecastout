import numpy as np
import pandas as pd
from forecastout.forecast_engine.anomaly_fixer import AnomalyFixer

def test_anomaly_fixer():
    # -- Logic
    anomaly_fixer = AnomalyFixer(
        df_forecast=predictions_input_df(),
        df_actual=actuals_input_df(),
        config=config_input_dict()
    )
    # -- Test
    pd.testing.assert_frame_equal(
        anomaly_fixer.df_forecast.copy(),
        output_df()
    )


def config_input_dict():
    config = {
        "horizon": 15,
        "months_to_backtest": 3,
        "models_to_use": ['autoarima', 'holtwinters', 'prophet'],
        "models_to_fix": ['seasonalnaive'],
        "models":
            {"seasonalnaive": {"freq": 12, "alpha_intervals": 95}}
    }
    return config



def actuals_input_df():
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


def predictions_input_df():
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
        {'forecast': [i * 5 for i in forecast_list],
         'forecast_lower': [i * 5 for i in forecast_lower_list],
         'forecast_upper': [i * 5 for i in forecast_upper_list],
         'model': model_list,
         'date': pd.to_datetime(date_list),
         }
    )


def output_df():
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01"
    ]
    forecast_list = [
        1.017697e+08,
        1.176161e+08,
        1.120136e+08,
        9.573360e+07,
        9.784148e+07,
        3.665602e+07
    ]
    forecast_lower_list = [
        8.603707e+07,
        1.018542e+08,
        9.583394e+07,
        7.955349e+07,
        8.166073e+07,
        2.247610e+07
    ]
    forecast_upper_list = [
        1.175023e+08,
        1.333779e+08,
        1.281933e+08,
        1.119138e+08,
        1.140222e+08,
        5.283758e+07
    ]
    model_list = [
        "autoarima/HW_A/HW_M/seasonalnaive",
        "autoarima/HW_A/HW_M/seasonalnaive",
        "autoarima/HW_A/HW_M/seasonalnaive",
        "autoarima/HW_A/HW_M/seasonalnaive",
        "autoarima/HW_A/HW_M/seasonalnaive",
        "autoarima/HW_A/HW_M/seasonalnaive"
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'forecast': forecast_list,
         'forecast_lower': forecast_lower_list,
         'forecast_upper': forecast_upper_list,
         'model': model_list,
         }
    )
