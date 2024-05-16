from forecastout.disaggregation_features.feature_creator \
    import FeatureCreator
import pandas as pd
import numpy as np


def test_feature_creator():
    feature_creator = FeatureCreator(df=input_df())
    feature_creator.get_day()
    feature_creator.get_month()
    feature_creator.get_year()
    feature_creator.get_day_of_week()
    feature_creator.get_holidays()
    pd.testing.assert_frame_equal(
        feature_creator.df,
        expected_df()
    )


def input_df():
    date_list = [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05",
        "2021-01-06",
        "2021-01-07",
        "2021-01-08",
        "2021-01-09",
        "2021-01-10",
        "2021-01-11",
        "2021-01-12",
        "2021-01-13",
        "2021-01-14",
        "2021-01-15",
        "2021-01-16",
        "2021-01-17",
        "2021-01-18",
        "2021-01-19",
        "2021-01-20"
    ]
    value_list = [
        24345.1108,
        31643.93512,
        37702.74413,
        63790.15493,
        49700.0231,
        42981.96545,
        59259.1285,
        56119.84923,
        43468.55374,
        45174.22881,
        69707.69637,
        62419.33631,
        63633.19097,
        60598.55433,
        56480.86634,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan
        ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list), 'value': value_list}
    )


def expected_df():
    date_list = [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05",
        "2021-01-06",
        "2021-01-07",
        "2021-01-08",
        "2021-01-09",
        "2021-01-10",
        "2021-01-11",
        "2021-01-12",
        "2021-01-13",
        "2021-01-14",
        "2021-01-15",
        "2021-01-16",
        "2021-01-17",
        "2021-01-18",
        "2021-01-19",
        "2021-01-20"
    ]
    value_list = [
        24345.1108,
        31643.93512,
        37702.74413,
        63790.15493,
        49700.0231,
        42981.96545,
        59259.1285,
        56119.84923,
        43468.55374,
        45174.22881,
        69707.69637,
        62419.33631,
        63633.19097,
        60598.55433,
        56480.86634,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan
        ]
    day_list = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20
    ]
    month_list = [1+i*0 for i in range(0, len(date_list))]
    year_list = [2021+i*0 for i in range(0, len(date_list))]
    day_of_week_list = [
        4,
        5,
        6,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
        1,
        2
    ]
    national_holiday_list = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    national_holiday_or_weekend_list = [
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0
    ]
    long_weekends_list = [
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0
    ]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'day': day_list,
         'month': month_list,
         'year': year_list,
         'day_of_week': day_of_week_list,
         'national_holiday': national_holiday_list,
         'national_holiday_or_weekend': national_holiday_or_weekend_list,
         'long_weekends': long_weekends_list}
    )
