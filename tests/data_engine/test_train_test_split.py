from forecastout.data_engine.train_test_split import TrainTestSplit
import pandas as pd
import numpy as np

CURRENT_MONTH = "2021-12-01"
MONTHS_TO_BACKTEST = 3
GRANULARITY_MONTH = True

def test_split_by_current_closed_month():
    pd.testing.assert_frame_equal(
        TrainTestSplit.split_by_current_closed_month(
            df=input_df(),
            current_month=CURRENT_MONTH,
            granularity_month=GRANULARITY_MONTH),
        expected_df_split_by_current_closed_month()
    )

def test_split_for_ts_backtest():
    pd.testing.assert_frame_equal(
        TrainTestSplit.split_for_ts_backtest(
            df=input_df(),
            current_month=CURRENT_MONTH,
            months_to_backtest=MONTHS_TO_BACKTEST),
        expected_df_split_for_ts_backtest()
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

    ]
    value_list = [
        5.830303e+06,
        7.400458e+06,
        7.449702e+06,
        6.480788e+06,
        6.995587e+06,
        1.675807e+06,
        8.372645e+06,
        1.132437e+07,
        0,
        4.509435e+06,
        1.331609e+07,
        1.282667e+07,
        np.nan,
        np.nan,
        np.nan
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )



def expected_df_split_by_current_closed_month():
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

    ]
    value_list = [
        5.830303e+06,
        7.400458e+06,
        7.449702e+06,
        6.480788e+06,
        6.995587e+06,
        1.675807e+06,
        8.372645e+06,
        1.132437e+07,
        0,
        4.509435e+06,
        1.331609e+07,
        1.282667e+07,
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
        1,
        1,
        1
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'test': test_list}
    )

def expected_df_split_for_ts_backtest():
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
        "2021-12-01"

    ]
    value_list = [
        5.830303e+06,
        7.400458e+06,
        7.449702e+06,
        6.480788e+06,
        6.995587e+06,
        1.675807e+06,
        8.372645e+06,
        1.132437e+07,
        0,
        4.509435e+06,
        1.331609e+07,
        1.282667e+07
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
        1,
        1,
        1
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'test': test_list}
    )