import pandas as pd
from forecastout.disaggregation_features.feature_normalizer \
    import FeatureNormalizer


def test_feature_normalizer():
    pd.testing.assert_frame_equal(
        FeatureNormalizer.normalization_base100(
             df=input_df(),
             norm_level=["month"],
             norm_feats=["value"]
         ),
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
        "2021-01-15"
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
        56480.86634
        ]
    month_list = [1+i*0 for i in range(0, len(date_list))]
    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'month': month_list
         }
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
        "2021-01-15"
    ]
    value_list = [
        34.924567,
        45.395181,
        54.086917,
        91.510921,
        71.297756,
        61.660287,
        85.010883,
        80.507393,
        62.358328,
        64.805224,
        100.000000,
        89.544397,
        91.285746,
        86.932373,
        81.025295
        ]
    month_list = [1+i*0 for i in range(0, len(date_list))]

    return pd.DataFrame(
        {'date': pd.to_datetime(date_list),
         'value': value_list,
         'month': month_list}
    )
