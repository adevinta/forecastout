import pandas as pd
from forecastout.disaggregation_engine\
    .remake_monthly_forecast_current_month \
    import remake_monthly_forecast_current_month


def test_remake_monthly_forecast_current_month():
    pd.testing.assert_frame_equal(
        remake_monthly_forecast_current_month(
            df_daily_forecast=input_df_daily_forecast(),
            df_actuals=input_df_actuals(),
            sum_aggregation=True
        ),
        expected_df()
    )

def input_df_actuals():
    date_list = ["2020-12-01", "2021-01-02", "2021-01-03"]
    value_list = [5, 10, 8]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )

def input_df_daily_forecast():
    date_list = ["2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01"]
    forecast_list = [7, 10, 11, 3]
    forecast_upper_list = [i + 5 for i in forecast_list]
    forecast_lower_list = [i - 1 for i in forecast_list]
    model_list = ['test_model', 'test_model', 'test_model', 'test_model']
    return pd.DataFrame({
        'date': pd.to_datetime(date_list),
        'forecast': forecast_list,
        'forecast_upper': forecast_upper_list,
        'forecast_lower': forecast_lower_list,
        'model': model_list
    })

def expected_df():
    date_list = ["2021-01-01", "2021-02-01"]
    forecast_list = [46, 3]
    forecast_lower_list = [43, 2]
    forecast_upper_list = [61, 8]
    model_list = ['test_model', 'test_model']
    return pd.DataFrame({
        'forecast': forecast_list,
        'forecast_lower': forecast_lower_list,
        'forecast_upper': forecast_upper_list,
        'date': pd.to_datetime(date_list),
        'model': model_list
    })
