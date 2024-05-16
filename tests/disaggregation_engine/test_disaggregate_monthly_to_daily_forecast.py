import pandas as pd
from forecastout.disaggregation_engine\
    .disaggregate_monthly_to_daily_forecast \
    import disaggregate_monthly_to_daily_forecast


def test_disaggregate_monthly_to_daily_forecast():
    pd.testing.assert_frame_equal(
        disaggregate_monthly_to_daily_forecast(
            df_daily_shares=input_df_daily_shares(),
            df_monthly_forecast=input_df_monthly_forecast()
        ),
        expected_df()
    )


def input_df_daily_shares():
    date_list = [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05"
    ]
    value_list = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.1
    ]
    return pd.DataFrame({'date': date_list, 'daily_share': value_list})


def input_df_monthly_forecast():
    date_list = [
        "2021-01-01"
    ]
    forecast_list = [
        100000
    ]
    forecast_upper_list = [
        150000
    ]
    forecast_lower_list = [
        75000
    ]
    model_list = [
        'test_model'
    ]
    return pd.DataFrame({
        'date': date_list,
        'forecast': forecast_list,
        'forecast_upper': forecast_upper_list,
        'forecast_lower': forecast_lower_list,
        'model': model_list
    })


def expected_df():
    date_list = [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05"
    ]
    forecast_list = [
        10000.0,
        20000.0,
        30000.0,
        40000.0,
        10000.0
    ]
    forecast_lower_list = [
        7500.0,
        15000.0,
        22500.0,
        30000.0,
        7500.0
    ]
    forecast_upper_list = [
        15000.0,
        30000.0,
        45000.0,
        60000.0,
        15000.0
    ]
    model_list = [
        'test_model',
        'test_model',
        'test_model',
        'test_model',
        'test_model'
    ]
    return pd.DataFrame({
        'date': date_list,
        'forecast': forecast_list,
        'forecast_lower': forecast_lower_list,
        'forecast_upper': forecast_upper_list,
        'model': model_list
    })
