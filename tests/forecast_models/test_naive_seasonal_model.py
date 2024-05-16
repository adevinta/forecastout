from forecastout.forecast_models.naive_seasonal import NaiveSeasonalModel
import pandas as pd
import numpy as np

DICT_CONFIG = {"freq": 12, "alpha_intervals": 95}


def test_naive_seasonal_model():
    model = NaiveSeasonalModel(
        df_train_y=input_df()['value'],
        dict_config=DICT_CONFIG,
        series_train_dates=input_df()['date'])
    pd.testing.assert_frame_equal(
        model.do_forecast(input_list_dates())[['date', 'forecast', 'model']],
        expected_df()[['date', 'forecast', 'model']]
    )


def input_list_dates():
    return [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]


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
        "2023-12-01"
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
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )


def expected_df():
    forecast_list = [
        19823030.0,
        23546912.0,
        22349106.0
    ]
    forecast_lower_list = [
        7459281.0,
        11183163.0,
        9985357.0
    ]
    forecast_upper_list = [
        32186778.0,
        35910660.0,
        34712856.0
    ]
    model_list = [
        "seasonalnaive",
        "seasonalnaive",
        "seasonalnaive"
    ]
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'forecast': np.float32(forecast_list),
         'forecast_lower': np.float32(forecast_lower_list),
         'forecast_upper': np.float32(forecast_upper_list),
         'model': model_list
         }
    )
