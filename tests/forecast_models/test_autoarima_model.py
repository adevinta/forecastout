from forecastout.forecast_models.autoarima_model \
    import AutoArimaModel
import pandas as pd

DICT_CONFIG = {"seasonality": "True", "freq": 12, "alpha_intervals": 0.05}


def test_autoarima_model():
    model = AutoArimaModel(
        df_train_y=input_df_value()['value'],
        dict_config=DICT_CONFIG)
    pd.testing.assert_frame_equal(
        model.do_forecast(input_list_dates()),
        expected_df()
    )


def input_list_dates():
    return [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]


def input_df_value():
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
    return pd.DataFrame({'value': value_list})


def expected_df():
    forecast_list = [
        2.617895e+07,
        3.065635e+07,
        2.884046e+07
    ]
    forecast_lower_list = [
        2.169750e+07,
        2.615167e+07,
        2.400181e+07
    ]
    forecast_upper_list = [
        3.066039e+07,
        3.516102e+07,
        3.367910e+07
    ]
    model_list = [
        "autoarima",
        "autoarima",
        "autoarima"
    ]
    date_list = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01"
    ]
    return pd.DataFrame(
        {'forecast': forecast_list,
         'forecast_lower': forecast_lower_list,
         'forecast_upper': forecast_upper_list,
         'model': model_list,
         'date': pd.to_datetime(date_list)}
    )
