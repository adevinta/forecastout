from forecastout.disaggregation_engine.disaggregation_model_trainer \
    import DisaggregationModelTrainer
from forecastout.disaggregation_engine.disaggregation_model_predictor \
    import DisaggregationModelPredictor
from forecastout.disaggregation_engine.get_daily_shares import \
    get_daily_shares
import pandas as pd


def test_random_forest_model():
    disaggregation_model_trainer = DisaggregationModelTrainer(
        df_train=input_df().loc[input_df()['test'] == 0].copy(),
        config=config_input_dict()
    )
    disaggregation_model_trained = disaggregation_model_trainer.train(
        config_input_dict()["disaggregation_model_selected"]
    )
    disaggregation_model_predictor = DisaggregationModelPredictor(
        model=disaggregation_model_trained,
        df=input_df().copy(),
        config=config_input_dict()
    )
    df_daily_prediction = disaggregation_model_predictor.predict()
    df_daily_shares = get_daily_shares(
        df_daily_prediction=df_daily_prediction.copy(),
        sum_aggregation=True
    )
    pd.testing.assert_frame_equal(
        df_daily_prediction.reset_index(drop=True),
        expected_daily_prediction_df().reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        df_daily_shares.reset_index(drop=True),
        expected_disaggregation_df().reset_index(drop=True)
    )


def config_input_dict():
    config = {
        "disaggregation_features": [
            "feature_1",
            "feature_2",
            "feature_3"
        ],
        "disaggregation_model_selected": "random_forest",
        "disaggregation_models":
            {"random_forest": {
                "param_grid":
                    {'n_estimators': [150],
                     'max_features': [2],
                     'max_depth': [3]
                     },
                "scoring": 'neg_root_mean_squared_error',
                "n_splits": 5,
                "n_repeats": 3,
                "refit": True,
                "return_train_score": True
            }
            }
    }
    return config


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
    test_list = (
            [0 + i * 0 for i in range(len(date_list) - 3)] +
            [1 + i * 0 for i in range(3)]
    )
    value_list1 = [
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
        {'date': pd.to_datetime(date_list),
         'test': test_list,
         'value': value_list1,
         'feature_1': [value * 1.1 for value in value_list1],
         'feature_2': [value / 2 for value in value_list1],
         'feature_3': [value ** 3 for value in value_list1]}
    )


def expected_daily_prediction_df():
    forecast_list = [
        25397864.73239,
        9952464.80652,
        25397864.73239
    ]
    model_list = [
        "random_forest",
        "random_forest",
        "random_forest"
    ]
    date_list = [
        "2023-10-01",
        "2023-11-01",
        "2023-12-01"
    ]
    return pd.DataFrame({
        'forecast': forecast_list,
        'model': model_list,
        'date': pd.to_datetime(date_list)
    })


def expected_disaggregation_df():
    daily_share_list = [
        1.0,
        1.0,
        1.0
    ]
    date_list = [
        "2023-10-01",
        "2023-11-01",
        "2023-12-01"
    ]
    return pd.DataFrame({
        'date': pd.to_datetime(date_list),
        'daily_share': daily_share_list
    })
